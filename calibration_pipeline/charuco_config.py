"""Validated ChArUco board configuration and OpenCV object construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2

from .dataset_loader import load_json_document


@dataclass(frozen=True)
class CharucoBoardConfig:
    dictionary: str
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    first_marker_id: int = 0
    marker_count: int | None = None

    def __post_init__(self) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV was built without the aruco module")
        if not self.dictionary.startswith("DICT_") or not hasattr(
            cv2.aruco, self.dictionary
        ):
            raise ValueError(f"Unsupported ArUco dictionary: {self.dictionary}")
        if self.squares_x < 2 or self.squares_y < 2:
            raise ValueError("A ChArUco board must contain at least 2x2 squares")
        if self.square_length_m <= 0 or self.marker_length_m <= 0:
            raise ValueError("Board lengths must be positive")
        if self.marker_length_m >= self.square_length_m:
            raise ValueError("marker_length_m must be smaller than square_length_m")
        if self.first_marker_id < 0:
            raise ValueError("first_marker_id must be nonnegative")
        if self.marker_count is not None and self.marker_count <= 0:
            raise ValueError("marker_count must be positive when supplied")

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> "CharucoBoardConfig":
        if document.get("type") != "charuco":
            raise ValueError("Board config type must be 'charuco'")
        return cls(
            dictionary=str(document["dictionary"]),
            squares_x=int(document["squares_x"]),
            squares_y=int(document["squares_y"]),
            square_length_m=float(document["square_length_m"]),
            marker_length_m=float(document["marker_length_m"]),
            first_marker_id=int(document.get("first_marker_id", 0)),
            marker_count=(
                int(document["marker_count"])
                if document.get("marker_count") is not None
                else None
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "CharucoBoardConfig":
        document, _ = load_json_document(path)
        if not isinstance(document, Mapping):
            raise ValueError("ChArUco config must contain a JSON object")
        return cls.from_mapping(document)

    def create_dictionary(self):
        dictionary_id = getattr(cv2.aruco, self.dictionary)
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            return cv2.aruco.getPredefinedDictionary(dictionary_id)
        if hasattr(cv2.aruco, "Dictionary_get"):  # OpenCV 3/early 4
            return cv2.aruco.Dictionary_get(dictionary_id)
        raise RuntimeError("This OpenCV version cannot create an ArUco dictionary")

    def create_board(self):
        dictionary = self.create_dictionary()
        size = (self.squares_x, self.squares_y)
        if hasattr(cv2.aruco, "CharucoBoard"):
            board = cv2.aruco.CharucoBoard(
                size, self.square_length_m, self.marker_length_m, dictionary
            )
        elif hasattr(cv2.aruco, "CharucoBoard_create"):  # OpenCV 3/early 4
            board = cv2.aruco.CharucoBoard_create(
                self.squares_x,
                self.squares_y,
                self.square_length_m,
                self.marker_length_m,
                dictionary,
            )
        else:
            raise RuntimeError("This OpenCV version cannot create a ChArUco board")

        ids = board.getIds() if hasattr(board, "getIds") else board.ids
        if self.first_marker_id:
            desired_ids = ids + self.first_marker_id
            if hasattr(board, "setIds"):
                board.setIds(desired_ids)
            else:
                raise RuntimeError(
                    "Nonzero first_marker_id is unsupported by this OpenCV version"
                )
        actual_count = int(ids.size)
        if self.marker_count is not None and actual_count != self.marker_count:
            raise ValueError(
                f"Configured marker_count={self.marker_count}, board has {actual_count}"
            )
        return board

