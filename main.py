"""Video Matting Tool - command line entry point.

    python main.py process --input in.mp4 --output out_green.mp4 --model rvm --background green
    python main.py serve
    python main.py --help
"""

from app.cli.main import run

if __name__ == "__main__":
    run()
