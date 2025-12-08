# src/Parser.py
# THIS CODE WILL HANDLE THE PARSER OBJECT.
# THIS OBJECT WILL TAKE A DICT OF REGEXES
# AND WILL EXTRACT THE STUFF FROM TEXT USING
# THE REGEX INTO A DICT WHERE THE KEYS WILL
# LINE UP BETWEEN THE REGEX AND THE OUTPUT

from typing import Dict, List

from src.logging_utils import get_logger

logger = get_logger(__name__)


class Parser:
    """
    URL Parser for CSV-like lines where each line contains up to three
    comma-separated URLs in the order: code (GitHub), dataset (HF dataset),
    model (HF model). Empty fields are allowed and
    represented as empty strings.
    """

    def __init__(self, filepath: str):
        """
        Initialize the Parser object.

        Parameters
        ----------
        filepath : str
            Path to the file containing newline-delimited URLs.

        Notes
        -----
        The constructor compiles regular expressions for each
        category and immediately categorizes the provided URLs.
        """
        self.filepath = filepath
        self.lines = self._loadLines()
        self.groups: List[Dict[str, str]] = self._categorize()

    def _loadLines(self) -> List[str]:
        """
        Read raw lines from the provided text file.
        Keeps empty cells separated by commas.

        Returns
        -------
        list[str]
            List of non-empty, stripped lines from the file.
        """
        with open(self.filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        logger.info("Loaded %d non-empty line(s) from %s",
                    len(lines),
                    self.filepath)
        return lines

    def _categorize(self) -> List[Dict[str, str]]:
        """
        Parse each line and assign entries to categories in order:
        code then dataset then model. Empty fields remain empty strings.

        Returns
        -------
        list[dict[str, str]]
            A list where each element corresponds to a line in the input
            with keys: 'git_url', 'dataset_url', 'model_url'.

        """
        results: List[Dict[str, str]] = []
        for line in self.lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) > 3:
                parts = parts[:3]
            elif len(parts) < 3:
                parts += [""] * (3 - len(parts))

            cleaned = [p if p.startswith("http") else "" for p in parts]
            if not any(cleaned):
                continue

            git_url, dataset_url, model_url = cleaned
            group = {
                "git_url": git_url,
                "dataset_url": dataset_url,
                "model_url": model_url,
            }
            results.append(group)
            logger.debug("Parsed group %s", group)

        return results

    def getGroups(self) -> List[Dict[str, str]]:
        """
        Return parsed URL groups per input line.

        Returns
        -------
        list[dict[str, str]]
            List of dictionaries with keys
            'git_url', 'dataset_url', 'model_url'.
        """
        return self.groups
