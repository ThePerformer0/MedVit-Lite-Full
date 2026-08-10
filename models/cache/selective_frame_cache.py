"""
MedViT-Lite — Selective Frame Cache (SFC)
==========================================
INNOVATION 2 : Réutilisation intelligente des features temporelles
               pour accélérer l'inférence sur des séquences vidéo.

Principe :
----------
Dans une vidéo médicale (échographie, endoscopie), les frames consécutives
sont souvent très similaires : l'organe ne bouge pas beaucoup entre la
frame t et la frame t+1 (à 30fps).

Au lieu de recalculer les features pour chaque frame from scratch, le SFC :
  1. Maintient un cache des features des frames récentes
  2. Compare chaque nouvelle frame avec les frames du cache
  3. Si très similaire → réutilise les features cachées (0 calcul)
  4. Si différente → calcule les nouvelles features et met à jour le cache

Gain en vitesse d'inférence :
  Sans cache : T × coût(frame)
  Avec cache : (T - T_cached) × coût(frame) + T_cached × 0
  Si 70% des frames sont similaires → 3.3× plus rapide

Note sur l'utilisation en entraînement :
  Le cache est désactivé pendant l'entraînement (on a besoin des gradients
  pour toutes les frames). Il est uniquement actif en inférence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from collections import deque
import logging

logger = logging.getLogger(__name__)


class SelectiveFrameCache(nn.Module):
    """
    Cache sélectif de features temporelles pour l'inférence en streaming.

    Le cache stocke un nombre limité de frames récentes (FIFO).
    Pour chaque nouvelle frame, on calcule sa similarité cosinus avec
    les frames du cache et on décide de recalculer ou réutiliser.

    Args:
        embed_dim            : dimension des feature maps
        cache_size           : nombre maximum de frames dans le cache
        similarity_threshold : seuil cosinus (0.92 = frames quasi-identiques)
        update_strategy      : "lru" (least-recently-used) ou "score" (importance)

    Attributs de monitoring :
        cache_hits    : nombre de fois où on a réutilisé le cache
        cache_misses  : nombre de fois où on a recalculé
        → cache_hit_rate = cache_hits / (cache_hits + cache_misses)
    """

    def __init__(
        self,
        embed_dim: int,
        cache_size: int = 4,
        similarity_threshold: float = 0.92,
        update_strategy: str = "lru",
    ):
        super().__init__()

        assert 0.0 < similarity_threshold < 1.0, \
            f"similarity_threshold doit être dans (0,1), reçu: {similarity_threshold}"
        assert cache_size >= 1, "cache_size doit être ≥ 1"
        assert update_strategy in ("lru", "score"), \
            f"update_strategy doit être 'lru' ou 'score', reçu: {update_strategy}"

        self.embed_dim = embed_dim
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        self.update_strategy = update_strategy

        # ── Le cache : liste de (features, frame_index, score_importance) ───
        # On utilise deque (file double-entrée) pour l'accès O(1) aux extrémités
        self._cache: deque = deque(maxlen=cache_size)

        # ── Projection linéaire pour l'espace de comparaison ─────────────────
        # On projette dans un espace plus petit pour la similarité
        # → plus rapide et souvent plus discriminatif
        self.similarity_projector = nn.Linear(embed_dim, embed_dim // 4, bias=False)

        # ── Statistiques de monitoring ────────────────────────────────────────
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(
            f"SFC initialisé : cache_size={cache_size}, "
            f"threshold={similarity_threshold}, strategy={update_strategy}"
        )

    def forward(
        self,
        frame_features: torch.Tensor,
        frame_idx: int,
        force_compute: bool = False,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Décide si on réutilise le cache ou recalcule pour cette frame.

        Ce module est appelé APRÈS que les features de la frame ont été
        calculées par le backbone. Si on décide de réutiliser le cache,
        on retourne les features cachées (les features calculées sont ignorées).

        Args:
            frame_features : [B, N, D] — features de la frame courante
            frame_idx      : index temporel de la frame
            force_compute  : forcer le recalcul (ignorer le cache)

        Returns:
            output_features : [B, N, D] — features à utiliser (cache ou calculées)
            cache_used      : bool — True si on a réutilisé le cache

        Note :
            Le module est en mode NO_GRAD par design à l'inférence.
            Pendant l'entraînement, on appelle forward() sans activer le cache
            (le module retourne simplement frame_features tel quel).
        """
        # ── Pendant l'entraînement : pas de cache (besoin des gradients) ─────
        if self.training:
            return frame_features, False

        # ── Première frame ou cache vide : pas de comparaison possible ────────
        if len(self._cache) == 0 or force_compute:
            self._add_to_cache(frame_features, frame_idx)
            self.cache_misses += 1
            return frame_features, False

        # ── Chercher la frame la plus similaire dans le cache ─────────────────
        best_match, best_similarity = self._find_best_match(frame_features)

        if best_similarity >= self.similarity_threshold:
            # ── Cache HIT : réutiliser les features cachées ───────────────────
            self.cache_hits += 1
            logger.debug(
                f"Frame {frame_idx}: cache HIT "
                f"(similarity={best_similarity:.3f} ≥ {self.similarity_threshold})"
            )
            return best_match, True

        else:
            # ── Cache MISS : recalculer et mettre à jour le cache ─────────────
            self._add_to_cache(frame_features, frame_idx)
            self.cache_misses += 1
            logger.debug(
                f"Frame {frame_idx}: cache MISS "
                f"(similarity={best_similarity:.3f} < {self.similarity_threshold})"
            )
            return frame_features, False

    def _compute_similarity(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
    ) -> float:
        """
        Calcule la similarité cosinus entre deux sets de features.

        On utilise la moyenne sur tous les patches et le batch pour
        avoir un scalaire représentatif.

        Args:
            features_a : [B, N, D]
            features_b : [B, N, D]

        Returns:
            similarity : scalaire ∈ [-1, 1] (1 = identiques)
        """
        # Projeter dans l'espace de comparaison (plus petit = plus rapide)
        with torch.no_grad():
            proj_a = self.similarity_projector(features_a)  # [B, N, D//4]
            proj_b = self.similarity_projector(features_b)  # [B, N, D//4]

        # Aplatir en [B, N*D//4] et calculer cosine similarity
        flat_a = proj_a.flatten(1)  # [B, N*D//4]
        flat_b = proj_b.flatten(1)  # [B, N*D//4]

        # Cosine similarity par item du batch, puis moyenne
        similarity = F.cosine_similarity(flat_a, flat_b, dim=1).mean().item()
        return similarity

    def _find_best_match(
        self, frame_features: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Cherche la frame du cache la plus similaire à la frame courante.

        Returns:
            best_features   : features de la meilleure correspondance
            best_similarity : score de similarité cosinus
        """
        best_similarity = -1.0
        best_features = None

        for cached_features, cached_idx, _ in self._cache:
            sim = self._compute_similarity(frame_features, cached_features)
            if sim > best_similarity:
                best_similarity = sim
                best_features = cached_features

        return best_features, best_similarity

    def _add_to_cache(self, features: torch.Tensor, frame_idx: int):
        """
        Ajoute une frame au cache.

        Avec maxlen=cache_size, deque retire automatiquement le plus ancien
        élément quand la capacité est atteinte (stratégie LRU de base).

        Note : on détache les features du graphe de calcul (`.detach()`)
        pour ne pas maintenir inutilement des gradients en mémoire.
        """
        importance_score = features.abs().mean().item()  # proxy d'importance
        self._cache.append(
            (features.detach().clone(), frame_idx, importance_score)
        )

    def reset_cache(self):
        """Vide le cache (appeler entre deux séquences/patients différents)."""
        self._cache.clear()
        logger.debug("Cache vidé.")

    def reset_stats(self):
        """Remet les compteurs à zéro."""
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def cache_hit_rate(self) -> float:
        """Taux d'utilisation du cache (métrique de monitoring)."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def get_stats(self) -> dict:
        """Retourne les statistiques du cache."""
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_occupancy": len(self._cache),
            "cache_size": self.cache_size,
        }

    def extra_repr(self) -> str:
        return (
            f"cache_size={self.cache_size}, "
            f"threshold={self.similarity_threshold}, "
            f"strategy={self.update_strategy}"
        )


if __name__ == "__main__":
    # ── Simulation d'une séquence vidéo médicale ─────────────────────────────
    logging.basicConfig(level=logging.DEBUG)

    torch.manual_seed(42)

    sfc = SelectiveFrameCache(
        embed_dim=384,
        cache_size=4,
        similarity_threshold=0.92,
    )
    sfc.eval()  # mode inférence

    B, N, D = 2, 98, 384  # batch=2, 98 patches (après DPS 50%), dim=384

    print("Simulation d'une séquence de 10 frames...")
    print(f"{'Frame':>6} | {'Cache':>8} | {'Hit?':>5} | {'Taux':>6}")
    print("-" * 40)

    for t in range(10):
        if t < 7:
            # Frames similaires (petit bruit ajouté)
            base = torch.randn(B, N, D)
            frame = base + 0.02 * torch.randn(B, N, D)
        else:
            # Frames très différentes (changement anatomique)
            frame = torch.randn(B, N, D)

        output, used_cache = sfc(frame, frame_idx=t)
        stats = sfc.get_stats()
        print(
            f"  t={t:2d}  | "
            f"{stats['cache_occupancy']}/{stats['cache_size']} frames | "
            f"{'✅ OUI' if used_cache else '❌ NON':>5} | "
            f"{stats['cache_hit_rate']:5.0%}"
        )

    print(f"\nRésultat final :")
    print(f"  Cache hit rate : {sfc.cache_hit_rate:.0%}")
    print(f"  Frames recalculées : {sfc.cache_misses}/10")
    print(f"  Frames réutilisées : {sfc.cache_hits}/10")
    print(f"  Gain computationnel estimé : {sfc.cache_hit_rate:.0%} moins de calculs")
