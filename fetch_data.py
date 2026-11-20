"""Download the real NREL/NLR ESIF data-center dataset into ./data/raw (run once; app can also do this)."""
import data as D

if __name__ == "__main__":
    D.download_esif(progress=lambda f, m: print(f"\r{f*100:5.1f}%  {m}", end="", flush=True))
    print("\nSaved to", D.data_dir() / "raw")
