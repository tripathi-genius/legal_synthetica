# How to Download the Fine-Tuned Models

We host our fine-tuned models on Google Drive. To easily download them directly to your server or local environment via the command line, we recommend using `gdown`.

## Step 1: Install gdown

If you do not already have `gdown` installed in your Python environment, you can install it using pip:

```
pip install gdown

```

## Step 2: Download the Models

Run the following commands in your terminal to pull the model files. Using `gdown` automatically handles the Google Drive virus-scan warnings for large files.

**Llama 3B**

```
gdown "https://drive.google.com/uc?id=1ISkz2vTCUlqJi71KIHotmCpBsbR_laEC"

```

**Llama 8B**

```
gdown "https://drive.google.com/uc?id=1eoxt-ZmehJDPNoWe300ZEHr7qRMPozzB"

```

**Qwen 3B**

```
gdown "https://drive.google.com/uc?id=1YVlBxS6T7q9mLG9WcvKZ9ag-prAIj9rz"

```

> **Troubleshooting:** If the downloads fail for any reason, you can try passing the original sharing link directly with the `--fuzzy` flag, like this:
> `gdown --fuzzy "https://drive.google.com/file/d/1ISkz2vTCUlqJi71KIHotmCpBsbR_laEC/view?usp=sharing"`