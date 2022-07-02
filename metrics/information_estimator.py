from torch.utils.data import DataLoader
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from helpers.serialization import df_read_json

def estimate_information():
    model_name = "allenai/t5-small-next-word-generator-qoogle"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    def run_model(input_string, **generator_args):
        input_ids = tokenizer.encode(input_string, return_tensors="pt")
        res = model.generate(input_ids, **generator_args)
        output = tokenizer.batch_decode(res, skip_special_tokens=True)
        return output


    def test_summaries(path):
        meansum = df_read_json(path)
        correct = 0
        total = 0
        for review in meansum['text']:
            processed = ''
            for word in review.split():
                processed += word + ' '
                next_word = run_model(processed)[0]
                current_index = len(processed)
                if review[current_index:current_index+len(next_word)] == next_word:
                    correct += 1
                total += 1
        print('correct: ' + str(correct))
        print('total: ' + str(total))
        print('ratio: ' + str(correct/total))


    test_summaries('./my_datasets/meansum_summaries_trimmed.json')
    test_summaries('./my_datasets/real_pegasus_summaries.json')
