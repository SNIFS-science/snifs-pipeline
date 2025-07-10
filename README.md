# SNIFS Pipeline

Current todos:

1. Copy data files over into NERSC
2. Get a basic image job running via sbatch
3. Create a minimal worker, perhaps using https://gist.github.com/bjorhn/2037a580f57b78813a7caf4419e60cfe or https://linen.prefect.io/t/26765966/we-want-to-experiment-with-developing-a-custom-worker-how-ca or https://docs.prefect.io/contribute/develop-a-new-worker-type as a starting point and see if I can get ir running locally just as running the image without sbatch
4. Port this over to using sbatch
5. Deploy the current flow to prefect cloud
6. See if I can run local worker connecting to prefect cloud