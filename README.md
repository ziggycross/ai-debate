# AI Safety via Debate
## Could LLM debaters be inherently marxist?

<p><a href="https://ai-debate.streamlit.app" target="_blank"><img src="https://img.shields.io/badge/Streamlit%20Cloud-grey?logo=Streamlit&amp;logoColor=FF4B4B&amp;labelColor=grey&amp;color=FF4B4B" alt="Streamlit Cloud"></a>
<a href="https://www.kaggle.com/datasets/ziggycross/ai-debate" target="_blank"><img src="https://img.shields.io/badge/Kaggle-grey?logo=Kaggle&amp;logoColor=white&amp;labelColor=grey&amp;color=20BEFF" alt="Kaggle Dataset"></a>
<a href="https://github.com/ziggycross/ai-debate" target="_blank"><img src="https://img.shields.io/github/stars/ziggycross/ai-debate" alt="GitHub Repo"></a></p>

Does AI debate natually regress to specific 'high-yield' topics? While reading AI Safety via Debate (Irving, G., Christiano, P. and Amodei, D., 2018), I jokingly asked if all debating LLMs were inherently marxist, as it was entirely possible a sufficiently long debate would always return to class struggle, or another similar 'universal' problem.

Through this project I aimed to explore this idea further by creating a large dataset of AI debate transcripts and performing a semantic analysis using a Python library called BERTopic.

To start, I tasked Claude 3.5 Sonnet with creating a set of susbtantial debate topics (using an LLM here helped me avoid bias). The debates were then performed by GPT-4o mini using a LangChain script, with each lasting between 10 and 20 turns.

After generating the dataset I got to work in BERTopic. BERTopic works by creating embeddings for the input documents, and clustering those embeddings. The library supports many methods for embedding the input documents, and while I just used the defaults, a more complex embedding model might provide better results.
After BERTopic's clustering was complete, I was able to produce several distribution visualisations of topics and keywords in order to answer my original question - do AI debates eventually tend towards any specific topics?

Currently, the results are inconclusive, my dataset is simply not big enough. I still believe it's an interesting question though, and I have published my source code so that someone may continue this project.
Further study should:

- Work on a larger scale, analysing thousands of documents instead of hundreds
- Define a more scientifically robust set of rules for determining a significant outcome. While BERTopic provides meaningful sematic understanding, a more quantitative framework would be useful.
- Combine different models and sizes to see if the results change.
- Try a 'group debate' approach, using several domain/subject matter expert models which have been fine tuned on factual data.
- Use full conversations instead of previous turns as context. I avoided this because of it's polynomial token cost, but it would likely significantly improve debate quality.

You can try the debate generation and analysis web app I built for this project [here](https://ai-debate.streamlit.app), and download my dataset from Kaggle [here](https://www.kaggle.com/datasets/ziggycross/ai-debate-samples).
If you are interested in extending this project or evaluating my methods, the source code an be found on GitHub [here](https://github.com/ziggycross/ai-debate).

This project was developed during the June cohort of AI Safety Fundamentals' course in AI Alignment, funded by BlueDot Impact. I would highly recommend the course to anyone interested in AI Safety research, more info can be found [here](https://aisafetyfundamentals.com/alignment/).
