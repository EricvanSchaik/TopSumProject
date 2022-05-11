import torch
from transformers import pipeline
from datasets import load_dataset
import os

# %%
use_cuda = True

if use_cuda and torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

# cache_directory = "D:\\Users\\Eric_\\.cache\\"

# os.environ["TRANSFORMERS_CACHE"] = cache_directory
# os.environ["HD_DATASETS_CACHE"] = cache_directory

classifier = pipeline("sentiment-analysis")
summarizer = pipeline("summarization")

# %%
review_text = "2 issues - Once I turned on the circle apps and installed this case, my battery drained twice as fast " \
              "as usual. I ended up turning off the circle apps, which kind of makes the case just a case... with a " \
              "hole in it. Second, the wireless charging doesn't work. I have a Motorola 360 watch and a Qi charging " \
              "pad. The watch charges fine but this case doesn't. But hey, it looks nice. "

review_text2 = "I’m embarrassed to admit that until recently, I have had a very negative opinion about “selfie " \
               "sticks” aka “monopods” aka “narcissticks.” But having reviewed a number of them recently, " \
               "they’re growing on me. This one is pretty nice and simple to set up and with easy instructions " \
               "illustrated on the back of the box (not sure why some reviewers have stated that there are no " \
               "instructions when they are clearly printed on the box unless they received different packaging than I " \
               "did). Once assembled, the pairing via bluetooth and use of the stick are easy and intuitive. Nothing " \
               "to it.<br /><br />The stick comes with a USB charging cable but arrived with a charge so you can use " \
               "it immediately, though it’s probably a good idea to charge it right away so that you have no " \
               "interruption of use out of the box. Make sure the stick is switched to on (it will light up) and " \
               "extend your stick to the length you desire up to about a yard’s length and snap away.<br /><br />The " \
               "phone clamp held the phone sturdily so I wasn’t worried about it slipping out. But the longer you " \
               "extend the stick, the harder it is to maneuver. But that will happen with any stick and is not " \
               "specific to this one in particular.<br /><br />Two things that could improve this: 1) add the option " \
               "to clamp this in portrait orientation instead of having to try and hold the stick at the portrait " \
               "angle, which makes it feel unstable; 2) add the opening for a tripod so that this can be used to sit " \
               "upright on a table for skyping and facetime eliminating the need to hold the phone up with your hand, " \
               "causing fatigue.<br /><br />But other than that, this is a nice quality monopod for a variety of " \
               "picture taking opportunities.<br /><br />I received a sample in exchange for my honest opinion. "

summary = summarizer(review_text)[0]["summary_text"]
summary2 = summarizer(review_text2)[0]["summary_text"]
print(classifier([review_text, summary]))
print(classifier([review_text2, summary2]))

# %%
review_dataset = load_dataset("amazon_us_reviews", "Toys_v1_00")
print(review_dataset)
