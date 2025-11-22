from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from recruit_restaurant_visitor_forecasting.config import PROCESSED_DATA_DIR

app = typer.Typer()

def get_first_str_values(s: pd.Series, n: int, sep: str = ' ') -> pd.Series:
    return s.str.split(sep).str[:n].str.join(sep)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
