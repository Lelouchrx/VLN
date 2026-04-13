from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from torch.utils.data import Dataset

from ..extras import logging


if TYPE_CHECKING:
    from datasets import Dataset as HFDataset

    from .processor import DatasetProcessor


logger = logging.get_logger(__name__)


class LazyMapDataset(Dataset):
    r"""Map-style dataset that preprocesses samples lazily in __getitem__."""

    def __init__(
        self,
        dataset: "HFDataset",
        dataset_processor: "DatasetProcessor",
        max_fallback_tries: int = 64,
    ) -> None:
        self.dataset = dataset
        self.dataset_processor = dataset_processor
        self.max_fallback_tries = max(1, max_fallback_tries)

    def __len__(self) -> int:
        return len(self.dataset)

    def _to_batched_example(self, example: Mapping[str, Any]) -> dict[str, list[Any]]:
        return {k: [v] for k, v in example.items()}

    def _is_valid_processed(self, processed: Mapping[str, Any]) -> bool:
        if len(processed) == 0:
            return False

        for value in processed.values():
            if not isinstance(value, list) or len(value) == 0:
                return False

        return True

    def _to_single_example(self, processed: Mapping[str, list[Any]]) -> dict[str, Any]:
        return {k: v[0] for k, v in processed.items()}

    def _process_index(self, index: int) -> dict[str, Any] | None:
        raw_example = self.dataset[index]
        processed = self.dataset_processor.preprocess_dataset(self._to_batched_example(raw_example))
        if not self._is_valid_processed(processed):
            return None

        return self._to_single_example(processed)

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset_size = len(self.dataset)
        if dataset_size == 0:
            raise IndexError("The dataset is empty.")

        index = index % dataset_size
        max_tries = min(self.max_fallback_tries, dataset_size)
        for offset in range(max_tries):
            current_index = (index + offset) % dataset_size
            processed = self._process_index(current_index)
            if processed is not None:
                if offset > 0:
                    logger.warning_rank0_once(
                        "Encountered invalid samples in lazy map dataset, falling back to nearby valid samples."
                    )
                return processed

        raise RuntimeError(
            "Cannot preprocess valid samples with lazy_map_dataset. "
            "Please check data format or disable lazy_map_dataset."
        )
