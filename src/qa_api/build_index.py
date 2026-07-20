from .pipeline import NewsPipeline


if __name__ == "__main__":
    count = NewsPipeline().build_index()
    print(f"Indexed {count} news chunks")
