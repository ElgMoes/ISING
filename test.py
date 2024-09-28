from pydub import AudioSegment
from pydub.playback import play

sound = AudioSegment.from_file("assets/sounds/decide.mp3")
play(sound)