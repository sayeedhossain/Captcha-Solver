import os

from PIL import Image, ImageDraw, ImageFont
import random
import glob

OUTPUT_DIR = 'generated_captcha_images2/'

# if the output directory does not exist, create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

fonts = []
with open("fonts.txt") as f:
    content = f.readlines()

acceptable_fonts = set(x.strip() for x in content)


for filename in glob.iglob('/usr/share/fonts/truetype/**/*.ttf', recursive=True):
    if filename in acceptable_fonts:
        fonts.append(ImageFont.truetype(filename, 15))

words = set()
for i in range(0, 10000):
    word = ""
    for j in range(0, 5):
        word += random.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    if word in words:
        continue

    img = Image.new("1", (70, 40))
    img.paste((1), [0, 0, img.size[0], img.size[1]])

    d = ImageDraw.Draw(img)
    font = random.choice(fonts)
    d.text((1, 5), word, font=font, fill=(0))
    img.save(OUTPUT_DIR + word + '.png')
    words.add(word)
