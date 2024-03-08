from openai import OpenAI
import pandas as pd
import pickle

# Function to generate a balanced prompt with randomly selected demonstrations
# Boolq data got from https://github.com/google-research-datasets/boolean-questions
def generate_prompt(train_path, dev_path, num_demos=8):
    # Opens the Boolq data (json format)
    boolq_data = pd.read_json(path_or_buf=train_path, lines=True)
    boolq_test = pd.read_json(path_or_buf=dev_path, lines=True)
    
    # Randomly pick num_demos prompts out from boolQ (yes/no balanced)
    # Sample seed fixed
    num = (int)(num_demos/2)
    true_demo = boolq_data.loc[boolq_data['answer'] == True].sample(num, random_state=1)
    false_demo = boolq_data.loc[boolq_data['answer'] == False].sample(num, random_state=1)
    demos = pd.concat([true_demo, false_demo])

    # Shuffle
    demos = demos.sample(frac = 1).values.tolist()
    # Item 1: question; Item 2: title; Item 3: answer; Item 4: passage
    
    # Generate demonstrations from the train dataset
    demonstration = ''
    for idx in range(len(demos)):
        each_demo = demos[idx]
        demonstration += f'Example %d: [title: %s\npassage: %s\nquestion: %s\nanswer: %s]\n\n' % (idx+1, each_demo[1], each_demo[3], each_demo[0], each_demo[2])

    # Generate prompts from the dev dataset
    true_labels = []
    # Sample = 15 False + 15 True
    # Sample seed fixed
    prompts = boolq_test.sample(30, random_state=5).values.tolist()

    prompt = []
    for idx in range(len(prompts)):
        each_prompt = prompts[idx]
        prompt.append(f'Complete the answers for tasks below with True or False:\n Task %d: [title: %s\npassage: %s\nquestion: %s\nanswer: ]\n' % (idx+1, each_prompt[1], each_prompt[3], each_prompt[0]))
        true_labels.append(f'{each_prompt[2]}')
    
    return demonstration, prompt, true_labels

def GPT_response(demo, prompt, OPENAI_API_KEY):
    client = OpenAI(api_key = OPENAI_API_KEY)
    GPT_out = ''
    for i in range(len(prompt)):
        for chunk in client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=demo+prompt[i],
        temperature=0.9,
        top_p=0.5,
        stream=True
        ):
            GPT_out += chunk.choices[0].text
    
    # Save output as file
    pickle.dump(GPT_out, file = open("GPT_out.pickle", "wb"))
    return GPT_out

# Function to evaluate GPT-3.5 model output with the true_answers
def evaluate_on_boolq(GPT_outputs, true_labels):
    correct=0
    length = len(GPT_outputs)
    assert(length == len(true_labels))
    for i in range(length):
        if(GPT_outputs[i] == true_labels[i]):
            correct += 1
    acc = correct / length

    return acc

if __name__ == '__main__':
    train_path = 'train.jsonl'
    dev_path = 'dev.jsonl'
    demo, prompt, true_labels = generate_prompt(train_path=train_path,dev_path=dev_path, num_demos=8)

    # Personal API key
    OPENAI_API_KEY = 'Your key here'
    GPT_out = GPT_response(demo, prompt, OPENAI_API_KEY)

    GPT_out = pickle.load(open("GPT_out.pickle", "rb"))
    out = GPT_out.split("\n")
    # Take out the first empty element and convert the strings to bool
    out = out[1:]

    print(f'OpenAI Accuracy for Boolq is: %.3f.' % evaluate_on_boolq(out, true_labels))


    