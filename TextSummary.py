#-*- coding:utf-8 -*-
import openai
import re

openai.api_key = "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"

# 定义总结文本的函数
def summarize_text(text, max_tokens=120):
    prompt = "Please summarize the following text:\n\n" + text + "\n\nSummary:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.7,
    )
    summary = response.choices[0].text
    summary = re.sub('^[0-9]*[.、）]', '', summary)
    return summary

# 定义使用 openai 将英文文本转换为中文文本的函数
def translate_text(prompt, model="text-davinci-002", tokens=300, temperature=0.6):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )

    translated_text = response.choices[0].text.strip()
    return translated_text

# 测试函数
text = """
The day for the release of GPT-4 is getting closer.

GPT-3 was announced in May 2020, almost two years ago. It was released one year after GPT-2 — which was also released a year after the original GPT paper was published. If this trend were to hold across versions, GPT-4 should already be here. It’s not, but OpenAI’s CEO, Sam Altman, said a few months ago that GPT-4 is coming. Current estimates forecast the release date sometime in 2022, likely around July-August.

Despite being one of the most awaited AI news, there’s little public info about GPT-4: what it’ll be like, its features, or its abilities. Altman conducted a Q&A last year and gave a few hints on OpenAI’s ideas for GPT-4 (he asked participants to be discrete about the info, and I’ve been silent — but seven months is a reasonable time margin). One thing he said for sure is that GPT-4 won’t have 100T parameters, as I hypothesized in a previous article (such a big model will have to wait).

It’s been some time since OpenAI revealed any info on GPT-4. However, some novel trends that are getting huge traction in the field of AI, particularly in NLP, could give us clues on GPT-4. It’s possible to make a few acceptable predictions from what Altman said given the success of these approaches and OpenAI’s involvement. And surely, those go beyond the well-known — and tired — practice of making the models ever-larger.

Here are my predictions on GPT-4 given the info we have from OpenAI and Sam Altman, and the current trends and the state-of-the-art in language AI. (Explicitly or contextually, I’ll make it clear which are guesses and which are certainties.)

Model size: GPT-4 won’t be super big
GPT-4 won’t be the largest language model. Altman said it wouldn’t be much bigger than GPT-3. The model will be certainly big compared to previous generations of neural networks, but size won’t be its distinguishing feature. It’ll probably lie somewhere in between GPT-3 and Gopher (175B-280B).

And there’s a good reason for this decision.

Megatron-Turing NLG, built by Nvidia and Microsoft last year, held the title of the largest dense neural network at 530B parameters — already 3x larger than GPT-3 — until very recently (Google’s PaLM holds the title now at 540B). But remarkably, some smaller models that came after MT-NLG reached higher performance levels.

Bigger ≠ better.

The existence of smaller models that are better has two implications.

First, companies have realized that using model size as a proxy to improve performance isn’t the only way to do it — nor the best way. In 2020, OpenAI’s Jared Kaplan and colleagues concluded that performance improves the most when increases in compute budget are allocated mostly to scaling the number of parameters, following a power-law relationship. Google, Nvidia, Microsoft, OpenAI, DeepMind, and other companies developing language models took those guidelines at face value.

But MT-NLG, being as big as it is, isn’t the best in terms of performance. In fact, it isn’t the best at any single category benchmark. Smaller models, like Gopher (280B), or Chinchilla (70B) — barely a fraction of its size — are way better than MT-NLG across tasks.

It’s become clear model size isn’t the only factor to achieve greater language understanding, which leads me to the second implication.

Companies are starting to reject the “larger is better” dogma. Having more parameters is just one factor among many that can improve performance. And the collateral damage (e.g. carbon footprint, compute costs, or entry barriers) makes it one of the worst factors to consider — although very straightforward to implement. Companies will think twice before building a gigantic model when they can get similar — or better — results from a smaller one.

Altman said they weren’t focusing on making models extremely larger anymore, but on getting the best out of smaller models. OpenAI researchers were early advocates of the scaling hypothesis but may have now realized other unexplored paths can lead to improved models.

GPT-4 won’t be much larger than GPT-3, and those are the reasons. OpenAI will shift the focus toward other aspects — like data, algorithms, parameterization, or alignment — that could bring significant improvements more cleanly. We’ll have to wait to see the capabilities of a 100T-parameter model.

Optimality: Getting the best out of GPT-4
Language models suffer from one critical limitation when it comes to optimization. The training is so expensive that companies have to make trade-offs between accuracy and cost. This often results in models being notably underoptimized.

GPT-3 was only trained once despite some errors that in other cases would have led to re-training. OpenAI decided to not do it due to the unaffordable costs, preventing researchers to find the best set of hyperparameters for the model (e.g. learning rate, batch size, sequence length, etc).

Another consequence of high training costs is that analyses of model behavior are constricted. When Kaplan’s team concluded model size was the most relevant variable to improve performance, they weren’t factoring in the number of training tokens — that is, the amount of data the models were fed. Doing so would’ve required prohibitive amounts of computing resources.

Tech companies followed Kaplan’s conclusions because it was the best they had. Ironically, Google, Microsoft, Facebook, and others “wasted” millions on ever-larger models — generating vast amounts of pollution in the process — motivated precisely by economic restrictions.

Now, companies, with DeepMind and OpenAI leading the way, are exploring other approaches. They’re trying to find optimal models instead of just bigger ones.

Optimal parameterization
Last month, Microsoft and OpenAI proved GPT-3 could be further improved if they trained the model with optimal hyperparameters. They found that a 6.7B version of GPT-3 increased its performance so much that it was comparable to the original 13B GPT-3 model. Hyperparameter tuning — unfeasible for larger models — resulted in a performance increase equivalent to doubling the number of parameters.

They found a new parameterization (μP) in which the best hyperparameters for a small model were also the best for a larger one of the same family. μP allowed them to optimize models of arbitrary size for a tiny fraction of the training cost. The hyperparameters can then be transferred virtually costless to the larger model.

Optimal-compute models
A few weeks ago, DeepMind revisited Kaplan’s findings and realized that, contrary to what was believed, the number of training tokens influences performance as much as model size. They concluded that, as more compute budget is available, it should be equally allocated to scaling parameters and data. They proved their hypothesis by training Chinchilla, a 70B model (4 times smaller than Gopher, previous SOTA) with 4 times more data than all large language models since GPT-3 (1.4T tokens — from the typical 300B).

The results were clear-cut. Chinchilla “uniformly and significantly” outperformed Gopher, GPT-3, MT-NLG, and all other language models across many language benchmarks: Current models are undertrained and oversized.

Given that GPT-4 will be slightly larger than GPT-3, the number of training tokens it’d need to be compute-optimal (following DeepMind’s findings) would be around 5 trillion — an order of magnitude higher than current datasets. The number of FLOPs they’d need to train the model to reach minimal training loss would be around 10–20x larger than they used for GPT-3 (using Gopher’s compute budget as proxy).

Altman may have been referring to this when he said in the Q&A that GPT-4 will use way more compute than GPT-3.

OpenAI will surely implement optimality-related insights into GPT-4 — although to which extent isn’t predictable, as their budget is unknown. What’s certain is they’ll focus on optimizing other variables apart from model size. Finding the best set of hyperparameters and the optimal-compute model size and number of parameters could result in unbelievable improvements in all benchmarks. All predictions for language models will fall short if these approaches are combined into one model.

Altman also said that people wouldn’t believe how good models can be without making them bigger. He may be suggesting that scaling efforts are over for now.

Multimodality: GPT-4 will be a text-only model
The future of deep learning is multimodal models. Human brains are multisensory because we live in a multimodal world. Perceiving the world one mode at a time greatly limits AI’s ability to navigate or understand it.

However, good multimodal models are significantly harder to build than good language-only or vision-only models. Combining visual and textual info into a single representation is a daunting task. We have very limited notions of how our brain does it (not that the deep learning community is taking into account insights from the cognitive sciences on brain structure and functionality), so we don’t know how to implement it in neural networks.

Altman said in the Q&A that GPT-4 won’t be multimodal (like DALL·E or MUM), but a text-only model. My guess is they’re trying to reach the limits of language models, tweaking factors like model and dataset size before jumping to the next generation of multimodal AI.

Sparsity: GPT-4 will be a dense model
Sparse models that leverage conditional computation using different parts of the model to process different types of inputs have found great success recently. These models easily scale beyond the 1T-parameter mark without suffering from high computing costs, creating a seemingly orthogonal relationship between model size and compute budget. However, the benefits of MoE approaches diminish on very large models.

Given the history of OpenAI’s focus on dense language models, it’s reasonable to expect GPT-4 will also be a dense model. And given that Altman said GPT-4 won’t be much larger than GPT-3, we can conclude that sparsity is not an option for OpenAI — at least for now.

Sparsity, similar to multimodality, will most likely dominate the future generations of neural networks, given that our brain — AI’s inspiration — relies heavily on sparse processing.

Alignment: GPT-4 will be more aligned than GPT-3
OpenAI has put a lot of effort to tackle the AI alignment problem: How to make language models follow our intentions and adhere to our values — whatever that means exactly. It’s not just a difficult problem mathematically (i.e. how can we make AI understand what we want precisely?), but also philosophically (i.e. there isn’t a universal way to make AI aligned with humans, as the variability in human values across groups is huge — and often conflictive).

Yet, they made the first attempt with InstructGPT, which is a renewed GPT-3 trained with human feedback to learn to follow instructions (whether those are well-intended or not isn’t yet factored into the models).

The main breakthrough from InstructGPT is that, regardless of its results on language benchmarks, it’s perceived as a better model by human judges (who form a very homogeneous group of people — OpenAI employees and english-speaking people —, so we should be careful about extracting conclusions). This highlights the necessity to get over using benchmarks as the only metric to assess AI’s ability. How humans perceive the models could be as important, if not more.

Given Altman and OpenAI’s commitment toward a beneficial AGI, I’m confident GPT-4 will implement — and build upon — the findings they got from InstructGPT.

They’ll improve the way they aligned the model because it was limited to OpenAI employees and English-speaking labelers. True alignment should include groups with all kinds of provenance and features regarding gender, race, nationality, religion, etc. It’s a great challenge and any steps toward that goal are welcomed (although we should be wary to call it alignment when it isn’t for most people).

"""

summary = summarize_text(text)
print(summary)
print("-------------------------------------------------")
english_text = summary.replace('\n', ' ')

# 用“将以下英文翻译成中文：{english_text}”或类似的方式构建提示
prompt = f"将以下英文翻译成中文：{english_text}"
chinese_translation = translate_text(prompt)

print("原文（英文）：", english_text)
print("翻译（中文）：", chinese_translation)
