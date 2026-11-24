from bpetok.tokenizer import get_pair_counts, merge, Tokenizer, decode, encoder

def train(text, vocab_size):
    tokens = text.encode("utf-8")
    print(f"Length of raw token: {len(tokens)}.\n")
    ids = list(map(int, tokens))
    merges = {}

    for i in range(vocab_size - 256):
        counts = get_pair_counts(ids)
        if not counts:
            break
        
        pair = min(counts, key=lambda p: (-counts[p], p))
        new_id = 256 + i
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
    return merges, ids


text = """
Before the emergence of transformer-based models in 2017, some language models were considered large relative to the computational and data constraints of their time. In the early 1990s, IBM's statistical models pioneered word alignment techniques for machine translation, laying the groundwork for corpus-based language modeling. In 2001, a smoothed n-gram model, such as those employing Kneser–Ney smoothing, trained on 300 million words, achieved state-of-the-art perplexity on benchmark tests.[5] During the 2000s, with the rise of widespread internet access, researchers began compiling massive text datasets from the web ("web as corpus"[6]) to train statistical language models.[7][8]
Moving beyond n-gram models, researchers started in 2000 to use neural networks as language models.[9] Following the breakthrough of deep neural networks in image classification around 2012,[10] similar architectures were adapted for language tasks. This shift was marked by the refinement[11] of word embeddings (e.g., Word2Vec by Mikolov in 2013) and sequence-to-sequence (seq2seq) models using LSTM. In 2016, Google transitioned its translation service to neural machine translation (NMT), replacing statistical phrase-based models with deep recurrent neural networks. These early NMT systems used LSTM-based encoder-decoder architectures, as they preceded the invention of transformers.
An illustration of the main components of the transformer model from the original paper, where layers were normalized after (instead of before) multiheaded attention
At the 2017 NeurIPS conference, Google researchers introduced the transformer architecture in their landmark paper "Attention Is All You Need".[12] This paper's goal was to improve upon 2014 seq2seq technology, and was based mainly on the attention mechanism developed by Bahdanau et al. in 2014.[13][14] The following year in 2018, BERT was introduced and quickly became "ubiquitous".[15] Though the original transformer has both encoder and decoder blocks, BERT is an encoder-only model. Academic and research usage of BERT began to decline in 2023, following rapid improvements in the abilities of decoder-only models (such as GPT) to solve tasks via prompting.[16]
Although decoder-only GPT-1 was introduced in 2018, it was GPT-2 in 2019 that caught widespread attention because OpenAI claimed to have initially deemed it too powerful to release publicly, out of fear of malicious use.[17] GPT-3 in 2020 went a step further and as of 2025 is available only via API with no offering of downloading the model to execute locally. But it was the consumer-facing chatbot ChatGPT in late 2022 that received extensive media coverage and public attention by 2023.[18] The 2023 GPT-4 was praised for its increased accuracy and as a "holy grail" for its multimodal capabilities.[19] OpenAI did not reveal the high-level architecture and the number of parameters of GPT-4. The release of ChatGPT led to an uptick in LLM usage across several research subfields of computer science, including robotics, software engineering, and societal impact work.[16] In 2024, OpenAI released the reasoning model OpenAI o1, which generates long chains of thought before returning a final answer.[20] Many LLMs with parameter counts comparable to those of OpenAI's GPT series have been developed.[21]
Since 2022, open-source models (those with their source code and weights made publicly available) have been gaining popularity, especially at first with BLOOM and LLaMA, though both have restrictions on the field of use. Mistral AI's open-weight models Mistral 7B and Mixtral 8x7B have a more permissive Apache License.[citation needed] In January 2025, DeepSeek released DeepSeek R1, a 671-billion-parameter open-weight model that performs comparably to OpenAI o1 but at a much lower price per token for users.[22]
Since 2023, many LLMs have been trained to be multimodal, having the ability to also process or generate other types of data, such as images, audio, or 3D meshes.[citation needed]
Open-weight LLMs have become more influential since 2023. Per Vake et al. (2025), community-driven contributions to open-weight models improve their efficiency and performance via collaborative platforms such as Hugging Face.[23]
Beyond basic text generation, various techniques have been developed to extend LLM capabilities, including the use of external tools and data sources, improved reasoning on complex problems, and enhanced instruction-following or autonomy through prompting methods.

Prompt engineering
Main article: Prompt engineering
In 2020, OpenAI researchers demonstrated that their new model GPT-3 could understand what format to use given a few rounds of Q and A (or other type of task) in the input data as example. This technique, called few-shot prompting, allows LLMs to be adapted to any task without requiring fine-tuning.[1] Also in 2022, it was found that the base GPT-3 model can generate an instruction based on user input. The generated instruction along with user input is then used as input to another instance of the model under a "Instruction: [...], Input: [...], Output:" format. The other instance is able to complete the output and often produces the correct answer in doing so. The ability to "self-instruct" makes LLMs able to bootstrap themselves toward a correct answer.[49]

Dialogue processing (chatbot)
An LLM can be turned into a chatbot by specializing it for conversation. User input is prefixed with a marker such as "Q:" or "User:" and the LLM is asked to predict the output after a fixed "A:" or "Assistant:". This type of model became commercially available in 2022 with ChatGPT, a sibling model of InstructGPT fine-tuned to accept and produce dialog-formatted text based on GPT-3.5. It could similarly follow user instructions. Before the stream of User and Assistant lines, a chat context usually starts with a few lines of overarching instructions, from a role called "developer" or "system" to convey a higher authority than the user's input. This is called a "system prompt".[citation needed]

Retrieval-augmented generation
Retrieval-augmented generation (RAG) is an approach that integrates LLMs with document retrieval systems. Given a query, a document retriever is called to retrieve the most relevant documents. This is usually done by encoding the query and the documents into vectors, then finding the documents with vectors (usually stored in a vector database) most similar to the vector of the query. The LLM then generates an output based on both the query and context included from the retrieved documents.[50]

Tool use
See also: Agent harness
Tool use is a mechanism that enables LLMs to interact with external systems, applications, or data sources. It can allow LLMs to, for example, fetch real-time information from an API or to execute code. A program separate from the LLM watches the output stream of the LLM for a special tool-calling syntax. When these special tokens appear, the program calls the tool accordingly and feeds its output back into the LLM's input stream.[51]

Early tool-using LLMs were fine-tuned on the use of specific tools. But fine-tuning LLMs for the ability to read API documentation and call APIs correctly has greatly expanded the range of tools accessible to an LLM.[52][53]

Agency
Main article: AI agent
An LLM is typically not an autonomous agent by itself, as it lacks the ability to interact with dynamic environments, recall past behaviors, and plan future actions. But it can be transformed into an agent by adding supporting elements: the role (profile) and the surrounding environment of an agent can be additional inputs to the LLM, while memory can be integrated as a tool or provided as additional input. Instructions and input patterns are used to make the LLM plan actions and tool use is used to potentially carry out these actions.[54]

In the DEPS ("describe, explain, plan and select") method, an LLM is first connected to the visual world via image descriptions. It is then prompted to produce plans for complex tasks and behaviors based on its pretrained knowledge and the environmental feedback it receives.[55]

The Reflexion method constructs an agent that learns over multiple episodes. At the end of each episode, the LLM is given the record of the episode, and prompted to think up "lessons learned", which would help it perform better at a subsequent episode. These "lessons learned" are stored as a form of long-term memory and given to the agent in the subsequent episodes.[56]

Monte Carlo tree search can use an LLM as rollout heuristic. When a programmatic world model is not available, an LLM can also be prompted with a description of the environment to act as world model.[57]

Multiple agents with memory can interact socially.[58]

Chaining
Prompt chaining was introduced in 2022.[59] In this method, a user manually breaks a complex problem down into several steps. In each step, the LLM receives as input a prompt telling it what to do and some results from preceding steps. The result from one step is then reused in a next step, until a final answer is reached. The ability of an LLM to follow instructions means that even non-experts can write a successful collection of stepwise prompts given a few rounds of trial and error.[60][61]

A 2022 paper demonstrated a separate technique called chain-of-thought prompting, which makes the LLM break the question down autonomously. An LLM is given some examples where the "assistant" verbally breaks down the thought process before arriving at an answer. The LLM mimics these examples and also tries to spend some time generating intermediate steps before providing the final answer. This additional step elicited by prompting improves the correctness of the LLM on relatively complex questions. On math word questions, a prompted model can exceed even fine-tuned GPT-3 with a verifier.[62][63] Chain-of-thought can also be elicited by simply adding an instruction like "Let's think step by step" to the prompt, in order to encourage the LLM to proceed methodically instead of trying to directly guess the answer.[64]

Model-native reasoning
Main articles: Reasoning model and Reflection (artificial intelligence)
In late 2024, a new approach to LLM development emerged with "reasoning models".[65] These are trained to generate step-by-step analysis before producing final answers, enabling better results on complex tasks, for instance in mathematics, coding and logic.[66] OpenAI introduced this concept with their o1 model in September 2024, followed by o3 in April 2025. On the International Mathematics Olympiad qualifying exam problems, GPT-4o achieved 13% accuracy while o1 reached 83%.[67]

In January 2025, the Chinese company DeepSeek released DeepSeek-R1, a 671-billion-parameter open-weight reasoning model that achieved comparable performance to OpenAI's o1 while being significantly more cost-effective to operate. Unlike proprietary models from OpenAI, DeepSeek-R1's open-weight nature allowed researchers to study and build upon the algorithm, though its training data remained private.[68]

These reasoning models typically require more computational resources per query compared to traditional LLMs, as they perform more extensive processing to work through problems step by step.[67]

Forms of input and output
Multimodality
See also: Multimodal learning
Multimodality means having multiple modalities, where a "modality" refers to a type of input or output, such as video, image, audio, text, proprioception, etc.[69] For example, Google PaLM model was fine-tuned into a multimodal model and applied to robotic control.[70] LLaMA models have also been turned multimodal using the tokenization method, to allow image inputs,[71] and video inputs.[72] GPT-4o can process and generate text, audio and images.[73]

A common method to create multimodal models out of an LLM is to "tokenize" the output of a trained encoder. Concretely, one can construct an LLM that can understand images as follows: take a trained LLM, and take a trained image encoder 
E
. Make a small multilayer perceptron 
f
, so that for any image 
y
, the post-processed vector 
f
(
E
(
y
)
)
 has the same dimensions as an encoded token. That is an "image token". Then, one can interleave text tokens and image tokens. The compound model is then fine-tuned on an image-text dataset. This basic construction can be applied with more sophistication to improve the model. The image encoder may be frozen to improve stability.[74] This type of method, where embeddings from multiple modalities are fused and the predictor is trained on the combined embeddings, is called early fusion.

Another method, called intermediate fusion, involves each modality being first processed independently to obtain modality-specific representations; then these intermediate representations are fused together.[75] In general, cross-attention is used for integrating information from different modalities. As an example, the Flamingo model uses cross-attention layers to inject visual information into its pre-trained language model.[76]

Non-natural languages
LLMs can handle programming languages similarly to how they handle natural languages. No special change in token handling is needed as code, like human language, is represented as plain text. LLMs can generate code from problem statements or instructions written in natural language, a process sometimes called vibe coding. They can also describe code in natural language or translate it into other programming languages. They were originally used as a code completion tool, but advances have moved them towards automatic programming. Services such as GitHub Copilot offer LLMs specifically trained, fine-tuned, or prompted for programming.[77][78]

In mathematical applications, LLMs can generate formal proofs in formal languages for automated proof assistants.[79]

In computational biology, transformer-based architectures, such as DNA LLMs, have also proven useful in analyzing biological sequences: protein, DNA, and RNA. With proteins they appear able to capture a degree of "grammar" from the amino-acid sequence, by mapping that sequence into an embedding. On tasks such as structure prediction and mutational outcome prediction, a small model using an embedding as input can approach or exceed much larger models using multiple sequence alignments (MSA) as input.[80] ESMFold, Meta Platforms' embedding-based method for protein structure prediction, runs an order of magnitude faster than AlphaFold2 thanks to the removal of an MSA requirement and a lower parameter count due to the use of embeddings.[81] Meta hosts ESM Atlas, a database of 772 million structures of metagenomic proteins predicted using ESMFold.[82] An LLM can also design proteins unlike any seen in nature.[83] Nucleic acid models have proven useful in detecting regulatory sequences,[84] sequence classification, RNA-RNA interaction prediction, and RNA structure prediction.[85]
"""

# merges1 = train(text, 100)

# def trackTrain(text):
#     vocab = [1000, 512]
#     for i in vocab:
#         merges, ids = train(text, i)
#         print(type(merges))
#         print(f"Length of leftout pairs after merges: {len(merges)}.\n")
#         print(f"Length of token after merges: {len(ids)}.\n")
#         # print(merges)


# trackTrain(text)

merges, ids = train(text, 512)
tok = Tokenizer(merges)
# print(tok.vocab[256], tok.vocab[257], tok.vocab[258])

# print(decode(ids, tok.vocab))

print(encoder(text, merges))