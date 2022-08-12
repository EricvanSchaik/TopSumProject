from src.summarization.huggingface_summarizer import summarize_text
from transformers import pipeline

if __name__ == '__main__':
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    ARTICLE = """
    
    All I had was static. I sent it back no trouble doing this.  No trouble with receiving my refund.
    I bought this for my hard of hearing father to watch tv without having to blast it in the living room and so far, there's been no static. it hasn't run out of power and lasts a very long time. I have no complaints.
    I was very excited to purchase these well reviewed headphones.  Unfortunately, we did not have a good experience with them- static was a major problem and the range for tuning in the television station was small.  Sadly, the static and background noise were too loud and rendered dialogue impossible to hear.  We returned them immediately (I will say that Amazon customer service was excellent) as they did not perform as advertised.
    Most of the time the headphones are fine, if a bit bulky; however, at times, for no apparent reason, static hits--and it is so sudden and so loud that I worry for the hearing I still have. . .therefore, I don't use them as much as I'd like.<br /><br />Doris Bridges<br />Homestead, FL
    These are just aweful.  After about 4 mos, they won't hold onto your head. If you look down while wearing them, they will hit the floor. Also, if your audio signal goes quiet for more than about 30 seconds, you hear a little \\"click\\" then deafening static hiss.  Was very dissapointed.
    constant static
    I am very happy with my purchase.  The sound it's great and no static at all. Plus I could finally watch TV in my bedroom without keeping my husband up. I will recommend this product to a friend.
    I LOVE these headphones!  I enjoy them so much that after years of purchasing on Amazon, this is my FIRST review ever.  I am not a high tech person or super into audio.  I am someone that was looking to a pair of wireless headphones to watch TV or movies with at night.  Every pair I have had in the past  had static or horrible sound.  These are GREAT, very clear sound no static and are so easy to hook up!  I liked them so much I recently purchased a second pair, which by the way can work off the same transmitter so we can both watch with headphones!  I would comment on how great the price is, but I would buy them  at a higher price as well.  VERY HAPPY WITH MY PURCHASE!
    For a factory refurb this pair of headphones are amazing.  I've had no issues w/ any static and get amazing reception throughout 3 floor of my house....whether i plug it in to my ipod or tv, I have amazing sound quality to the point where it's distracting watching tv because i can hear some of the background sounds such as a clock ticking on the wall!  Can't say enough about them and highly recommend them.
    Did alot of searching and checking reviews. My husband uses headphones when he goes out in hottub and kept buying cheap pairs.  Was tried of hearing him complain about them and yes even throwing them.  Well these are the BEST!! He loves them grat sound no static and no he hasn't thrown them once. As a matter of fact he says I should of bought two so I could use other pair. Well I use his cause I get up a lot earlier. Shhhhhhh!  I am glad I did alot of searching for right pair.
    These headphones are all they say they are. Great sound and no static due to the awesome easy tuning. I use them while I'm on the treadmill and they tend to slip a bit. The range is exceptionally good and they receive well through walls.I wore them outside to the mailbox which is over 100 feet from the house and they still had great sound. The reason I gave it only 4 stars is that the tuning and volume dials are so close together. I guess it's just a matter of getting use to. Other than that it is well worth the  purchase.
    For me the set up easy, I thought I'd have to play with the frequency adj., but out of the box It so clear no static at all.<br />I'm thinking of buying another set. I had purchased a different type before and kept getting a clicking noise in the headphone unless I turned my head a certain way. I'd say buy'em they're super.

    """

    print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))

