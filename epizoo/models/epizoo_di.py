# epizoo/models/epizoo_di.py

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from epizoo.models.epizoo import EpiZoo, EpiZooConfig


SpeciesInput = Union[
    int,
    str,
    Sequence[Union[int, str]],
    torch.Tensor,
]


class EpiZooDI(EpiZoo):
    """
    EpiZoo model for data imputation.

    EpiZooDI reuses the full EpiZoo architecture and only replaces
    the signal reconstruction loss with MSE loss.

    Difference from EpiZoo:
        EpiZoo   : signal_loss_fn = BCEWithLogitsLoss
        EpiZooDI : signal_loss_fn = MSELoss
    """

    def __init__(self, cfg: Optional[EpiZooConfig] = None):
        super().__init__(cfg)

        # Data imputation predicts continuous TF-IDF signals.
        self.signal_loss_fn = nn.MSELoss()

    def compute_signal_loss(
        self,
        cell_emb: torch.Tensor,
        input_species: SpeciesInput,
        signals: Union[torch.Tensor, Dict[str, Optional[torch.Tensor]]],
    ):
        """
        Compute MSE signal reconstruction loss.

        This method supports both:

        1. Single-species DI style:
            input_species = "Human" / "Mouse" / 0 / 1
            signals = Tensor [batch_size, vocab_size]

        2. General EpiZoo style:
            input_species = list or tensor of species ids
            signals = {
                "human": signals_human,
                "mouse": signals_mouse,
            }
        """

        species = self._normalize_species(
            input_species=input_species,
            batch_size=cell_emb.size(0),
        )

        if isinstance(signals, torch.Tensor):
            signals = self._wrap_single_species_signals(
                species=species,
                signals=signals,
            )

        return super().compute_signal_loss(
            cell_emb=cell_emb,
            input_species=species,
            signals=signals,
        )

    def predict_signal(
        self,
        cell_emb: torch.Tensor,
        species: Union[int, str],
    ) -> torch.Tensor:
        """
        Predict reconstructed signals for a single species.

        This replaces the old `return_SD_output=True` logic.
        """

        species = self._normalize_one_species(species)
        return self.signal_decoder(cell_emb, species)

    @classmethod
    def _wrap_single_species_signals(
        cls,
        species: List[str],
        signals: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Convert a single signal tensor into the dictionary format expected by EpiZoo.
        """

        unique_species = set(species)

        if len(unique_species) != 1:
            raise ValueError(
                "Tensor `signals` is only supported for single-species batches. "
                "For mixed-species batches, pass signals as a dict."
            )

        sp = next(iter(unique_species))

        return {
            "human": signals if sp == "human" else None,
            "mouse": signals if sp == "mouse" else None,
        }

    @classmethod
    def _normalize_species(
        cls,
        input_species: SpeciesInput,
        batch_size: int,
    ) -> List[str]:
        """
        Normalize species input to a list of species names.

        Supports:
            0 / 1
            "human" / "mouse"
            "Human" / "Mouse"
            list or tensor of the above
        """

        if isinstance(input_species, torch.Tensor):
            input_species = input_species.detach().cpu().tolist()

        if isinstance(input_species, (int, str)):
            return [cls._normalize_one_species(input_species)] * batch_size

        species = [
            cls._normalize_one_species(x)
            for x in input_species
        ]

        if len(species) != batch_size:
            raise ValueError(
                "`input_species` length should match batch size. "
                f"Got len(input_species)={len(species)}, batch_size={batch_size}."
            )

        return species

    @staticmethod
    def _normalize_one_species(species: Union[int, str]) -> str:
        """
        Normalize one species id or name.
        """

        if species == 0:
            return "human"

        if species == 1:
            return "mouse"

        if isinstance(species, str):
            species = species.lower()

            if species == "human":
                return "human"

            if species == "mouse":
                return "mouse"

        raise ValueError(
            "Species should be 0, 1, 'human', 'mouse', 'Human', or 'Mouse'."
        )