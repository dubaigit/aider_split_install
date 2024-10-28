from setuptools import setup, find_packages

setup(
    name="aider-voice",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tkinter",
        "pyaudio",
        "websockets",
        "openai",
        "numpy",
        "pyperclip",
        "sounddevice",
    ],
    entry_points={
        "console_scripts": [
            "aider-voice=aider_wrapper:main",
        ],
    },
)
