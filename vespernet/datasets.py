from pathlib import Path
from typing import List, Optional, Tuple

from torch.utils.data import Dataset

from .utils import is_image_file, load_image


class LowLightDataset(Dataset):
    def __init__(
        self,
        mode: str,
        source_low_dir: Optional[str],
        source_high_dir: Optional[str],
        target_low_dir: Optional[str],
        resize: Optional[Tuple[int, int]] = None,
        file_list: Optional[str] = None,
    ) -> None:
        self.mode = mode
        self.resize = resize

        if file_list is not None:
            self.low_paths, self.high_paths = self._load_from_file_list(file_list, mode)
            return

        if mode == "source_paired":
            if not source_low_dir or not source_high_dir:
                raise ValueError("source_low_dir and source_high_dir are required for source_paired")
            self.low_paths, self.high_paths = self._build_paired_paths(source_low_dir, source_high_dir)
        elif mode == "paired_by_index":
            if not source_low_dir or not source_high_dir:
                raise ValueError("source_low_dir and source_high_dir are required for paired_by_index")
            self.low_paths, self.high_paths = self._build_paired_by_index_paths(source_low_dir, source_high_dir)
        elif mode == "source_low_only":
            if not source_low_dir:
                raise ValueError("source_low_dir is required for source_low_only")
            self.low_paths = self._build_single_paths(source_low_dir)
            self.high_paths = None
        elif mode == "target_low_only":
            if not target_low_dir:
                raise ValueError("target_low_dir is required for target_low_only")
            self.low_paths = self._build_single_paths(target_low_dir)
            self.high_paths = None
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def _load_from_file_list(file_list: str, mode: str) -> tuple[List[Path], Optional[List[Path]]]:
        file_list_path = Path(file_list)
        base = file_list_path.parent
        low_paths: List[Path] = []
        high_paths: Optional[List[Path]] = [] if mode != "source_low_only" else None
        with file_list_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                low_path = Path(parts[0])
                low_paths.append(low_path if low_path.is_absolute() else base / low_path)
                if high_paths is not None and len(parts) > 1:
                    high_path = Path(parts[1])
                    high_paths.append(high_path if high_path.is_absolute() else base / high_path)
        return low_paths, high_paths

    @staticmethod
    def _build_single_paths(dir_path: str) -> List[Path]:
        root = Path(dir_path)
        paths = [path for path in root.iterdir() if path.is_file() and is_image_file(path)]
        return sorted(paths)

    def _build_paired_paths(self, low_dir: str, high_dir: str) -> tuple[List[Path], List[Path]]:
        low_root = Path(low_dir)
        high_root = Path(high_dir)
        low_map = {path.stem: path for path in low_root.iterdir() if path.is_file() and is_image_file(path)}
        high_map = {path.stem: path for path in high_root.iterdir() if path.is_file() and is_image_file(path)}
        keys = sorted(set(low_map) & set(high_map))
        return [low_map[key] for key in keys], [high_map[key] for key in keys]

    def _build_paired_by_index_paths(self, low_dir: str, high_dir: str) -> tuple[List[Path], List[Path]]:
        low_paths = self._build_single_paths(low_dir)
        high_paths = self._build_single_paths(high_dir)
        count = min(len(low_paths), len(high_paths))
        return low_paths[:count], high_paths[:count]

    def __len__(self) -> int:
        return len(self.low_paths)

    def __getitem__(self, index: int) -> dict:
        low_path = self.low_paths[index]
        sample = {
            "low": load_image(str(low_path), resize=self.resize).float(),
            "name": low_path.name,
        }
        if self.high_paths is not None:
            high_path = self.high_paths[index]
            sample["high"] = load_image(str(high_path), resize=self.resize).float()
        return sample
