from pipeline import bluebikes_pipeline

if __name__ == "__main__":
    bluebikes_pipeline.deploy(
        name="bluebikes-local-deployment",
        work_pool_name="default"
    )
