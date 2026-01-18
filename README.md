# MelaNone

## Project Description
MelaNone is a limb-scanning system designed to help detect melanoma early on. A patient places their arm or leg into the scan chamber, where cameras capture photos of the limb from multiple angles. These images are then stitched together to form a 3D object of the limb.

At the same time, smaller segments of the captured images are streamed to a PyTorch AI melanoma model that detects skin spots then classifies them as likely benign or potentially concerning. Once the analysis has completed, the results are summarized and displayed on our website.

## Getting Started
To get started with MelaNoma, clone this repository and follow the installation instructions below:

```git clone https://github.com/jasonkwok475/MelaNone.git

pip install uv
uv venv
source .venv/Scripts/activate

cd frontend
npm run build

cd ..
uv run backend/server.py```

Now, you can visit the site at http://127.0.0.1:5000/ 


## Demo

https://youtube.com/shorts/z7EJEKKdNtc?si=PFJrTsuNlRYjEFIi