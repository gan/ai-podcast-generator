# Podcast Generator


### Installing ffmpeg

#### On Windows

1.  Download the latest version of ffmpeg from the  official website.
2.  Extract the downloaded file.
3.  Add the bin folder from the extracted file to your system's PATH.
   
Or with chocolatey

```bash
choco install ffmpeg
```

#### On macOS

You can install ffmpeg using Homebrew:
```bash
brew  install  ffmpeg
```
#### On Linux

You can install ffmpeg using apt:

```bash
sudo apt update
sudo apt install ffmpeg
```

After installing ffmpeg, you can run the application as described in the "How to Run" section.
## How to Run

1. Clone the repository.
2. Install the required Python libraries using conda:

3. Run the Streamlit application:

```bash
streamlit run app.py
```

4. Open the application in your web browser at `http://localhost:8501`.

## Usage

1. Select your host character and the guests you would like to have on the show.
2. Enter your podcast topic.
3. Set the number of dialogues you want in the podcast.
4. Define the persona for each character.
5. Click the "Submit" button to generate the podcast.
6. The application will generate a transcript and an audio file of the podcast.


