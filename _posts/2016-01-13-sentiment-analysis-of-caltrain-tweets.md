---
ID: 17
post_title: Sentiment Analysis of Caltrain Tweets
author: Ben Everson
post_date: 2016-01-13 19:07:49
post_excerpt: ""
layout: post
permalink: >
  http://techvault.svds.io/wordpress/sentiment-analysis-of-caltrain-tweets/
published: true
voting_upvote:
  - "1"
voting_downvote:
  - "1"
vortex_system_likes:
  - "0"
vortex_system_dislikes:
  - "0"
vortex_system_user_1:
  - 'a:2:{s:5:"liked";s:5:"liked";s:8:"disliked";s:8:"disliked";}'
"vortex_system_user_::1":
  - 'a:2:{s:5:"liked";s:5:"liked";s:8:"disliked";s:8:"disliked";}'
---
This notebook walks through an example sentiment analysis task using scikit-learn and pandas. Specifically, the task is to classify a test set of tweets according to the sentiment they express (positive or negative). To do this, a training set is built using the <a href="http://help.sentiment140.com/for-students">Sentiment 140</a> data set, and binary classification models are built and trained in scikit-learn. Specifically, we use scikit-learn's pipeline and grid search cross validation tools to compare multiple models rapidly. The results of classification are visualized with wordclouds.

&nbsp;
<div class="cell border-box-sizing text_cell rendered">
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1>Sentiment Analysis of Caltrain Tweets</h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

To detect the sentiment of incoming tweets about Caltrain, we need a model that can recognize positive and negative sentiment in documents (tweets, in our case). To train this model, we first need a dataset of tweets that have been pre-assigned a sentiment. Here we use the Sentiment140 dataset, found here: <a href="http://help.sentiment140.com/for-students/">http://help.sentiment140.com/for-students/</a>. This dataset consists of 1.6m tweets, each labeled with the corresponding sentiment (0 for negative and 4 for positive). In the Sentiment140 dataset, sentiment was automatically assigned by looking at the use of emoticons in the tweet text: those tweets containing negative emoticons are assigned a negative sentiment, and those with positive emoticons are assigned a postive sentiment.

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [16]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>

<span class="c"># read in the Sentiment140 dataset</span>
<span class="n">sentiment140</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s">'training.1600000.processed.noemoticon.csv'</span><span class="p">,</span> <span class="n">header</span><span class="o">=</span><span class="k">None</span><span class="p">,</span> <span class="n">encoding</span><span class="o">=</span><span class="s">"ISO-8859-1"</span><span class="p">)</span>
<span class="n">sentiment140</span><span class="o">.</span><span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s">'polarity'</span><span class="p">,</span> <span class="s">'tweet_id'</span><span class="p">,</span> <span class="s">'date'</span><span class="p">,</span> <span class="s">'query'</span><span class="p">,</span> <span class="s">'user'</span><span class="p">,</span> <span class="s">'text'</span><span class="p">]</span>

<span class="c"># re-format the sentiment to allow for binary classification</span>
<span class="n">sentiment140</span><span class="p">[</span><span class="s">'sentiment'</span><span class="p">]</span> <span class="o">=</span> <span class="n">sentiment140</span><span class="p">[</span><span class="s">'polarity'</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">pol</span><span class="p">:</span> <span class="mi">0</span> <span class="k">if</span> <span class="n">pol</span> <span class="o">==</span> <span class="mi">0</span> <span class="k">else</span> <span class="mi">1</span><span class="p">)</span>

<span class="c"># show the first few lines</span>
<span class="n">sentiment140</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[16]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<table class="dataframe" border="1">
<thead>
<tr>
<th></th>
<th>polarity</th>
<th>tweet_id</th>
<th>date</th>
<th>query</th>
<th>user</th>
<th>text</th>
<th>sentiment</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>0</td>
<td>1467810369</td>
<td>Mon Apr 06 22:19:45 PDT 2009</td>
<td>NO_QUERY</td>
<td>_TheSpecialOne_</td>
<td>@switchfoot http://twitpic.com/2y1zl - Awww, t...</td>
<td>0</td>
</tr>
<tr>
<th>1</th>
<td>0</td>
<td>1467810672</td>
<td>Mon Apr 06 22:19:49 PDT 2009</td>
<td>NO_QUERY</td>
<td>scotthamilton</td>
<td>is upset that he can't update his Facebook by ...</td>
<td>0</td>
</tr>
<tr>
<th>2</th>
<td>0</td>
<td>1467810917</td>
<td>Mon Apr 06 22:19:53 PDT 2009</td>
<td>NO_QUERY</td>
<td>mattycus</td>
<td>@Kenichan I dived many times for the ball. Man...</td>
<td>0</td>
</tr>
<tr>
<th>3</th>
<td>0</td>
<td>1467811184</td>
<td>Mon Apr 06 22:19:57 PDT 2009</td>
<td>NO_QUERY</td>
<td>ElleCTF</td>
<td>my whole body feels itchy and like its on fire</td>
<td>0</td>
</tr>
<tr>
<th>4</th>
<td>0</td>
<td>1467811193</td>
<td>Mon Apr 06 22:19:57 PDT 2009</td>
<td>NO_QUERY</td>
<td>Karoli</td>
<td>@nationwideclass no, it's not behaving at all....</td>
<td>0</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

The dataset contains 6 fields:

<strong><code>polarity</code></strong> = the assigned sentiment of the tweet (0 - negative, 4 - positive)

<strong><code>tweet-id</code></strong> = unique id for the tweet

<strong><code>date</code></strong> = the timestamp for the tweet

<strong><code>user</code></strong> = twitter handle who sent the tweet

<strong><code>text</code></strong> = the text of the tweet

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [17]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c"># determine the proportion of positive and negative tweets in the dataset</span>
<span class="n">negative_tweets</span> <span class="o">=</span> <span class="n">sentiment140</span><span class="p">[</span><span class="n">sentiment140</span><span class="p">[</span><span class="s">'sentiment'</span><span class="p">]</span> <span class="o">==</span> <span class="mi">0</span><span class="p">]</span>
<span class="n">positive_tweets</span> <span class="o">=</span> <span class="n">sentiment140</span><span class="p">[</span><span class="n">sentiment140</span><span class="p">[</span><span class="s">'sentiment'</span><span class="p">]</span> <span class="o">==</span> <span class="mi">1</span><span class="p">]</span>
<span class="nb">print</span><span class="p">((</span><span class="s">"There are {} positive and {} negative tweets in the dataset."</span><span class="p">)</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">positive_tweets</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> 
                                                                              <span class="n">negative_tweets</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]))</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>There are 800000 positive and 800000 negative tweets in the dataset.
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

This dataset contains an even number of positive and negative tweets.

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3>Train-Test-Split</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

Before any models are trained, it is best to split the available data into training and test sets. <strong><code>scikit-learn</code></strong> includes a method for automatically splitting the data into randomly shuffled training and test sets. In the code below, we hold out 20% of the data as a test set.

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [18]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c"># split the data into training and test sets</span>
<span class="kn">from</span> <span class="nn">sklearn.cross_validation</span> <span class="k">import</span> <span class="n">train_test_split</span>
<span class="n">tweets_train</span><span class="p">,</span> <span class="n">tweets_test</span><span class="p">,</span> <span class="n">labels_train</span><span class="p">,</span> <span class="n">labels_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">sentiment140</span><span class="p">[</span><span class="s">'text'</span><span class="p">],</span> 
                                                                        <span class="n">sentiment140</span><span class="p">[</span><span class="s">'sentiment'</span><span class="p">],</span> 
                                                                        <span class="n">train_size</span><span class="o">=.</span><span class="mi">80</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2>1. Creating Feature Representations Of Tweets</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3>Tweets to Vectors of Word Counts</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

The first step in creating feature representations of tweets is to transform each tweet into a vector of word count. Each dimension in this vector corresponds to a unique word in the vocabularity of the entire corpus. <strong><code>scikit-learn</code></strong> offers a handy helper object for this exact purpose: <strong><code>CountVectorizer</code></strong>. This object can accomodate a list (or similar iterable) of strings and will produce the corresponding feature matrix in sparse matrix format.

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [19]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="kn">from</span> <span class="nn">sklearn.feature_extraction.text</span> <span class="k">import</span> <span class="n">CountVectorizer</span>
<span class="n">vectorizer</span> <span class="o">=</span> <span class="n">CountVectorizer</span><span class="p">()</span>
<span class="n">vectorized_tweets</span> <span class="o">=</span> <span class="n">vectorizer</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">tweets_train</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3>Word Counts to Word Frequency</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

The next step in creating robust feature representations of tweets is to normalize the results of the above transformation. One scheme for doing this is to convert the raw counts that appear at each element of the vector to tf-idf scores. In essence, this technique gives a high score to any word that appears often in the tweet but not often throughtout the corpus of tweets. In this sense, these scores caputre the information content of the words in the tweet, not just their occurence. <strong><code>scikit-learn</code></strong> again offfers a simple interface for transforming a vector of counts to a vector of tf-idf scores: <strong><code>TfidfTransformer</code></strong>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [20]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="kn">from</span> <span class="nn">sklearn.feature_extraction.text</span> <span class="k">import</span> <span class="n">TfidfTransformer</span>
<span class="n">transformer</span> <span class="o">=</span> <span class="n">TfidfTransformer</span><span class="p">(</span><span class="n">use_idf</span><span class="o">=</span><span class="k">True</span><span class="p">)</span>
<span class="n">transformed_tweets</span> <span class="o">=</span> <span class="n">transformer</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">vectorized_tweets</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2>2. Training a Classification Model</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

Since we have a corresponding sentiment for each tweet in the Sentiment140, we have everything we need to build a classification model. One simple model, which is very well suited for the task of document classification in this fashion is the <strong>Multinomial Naive Bayes</strong> classifier. <strong><code>scikit-learn</code></strong> has an implementation of this classifier: <strong><code>MultinomialNB</code></strong>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [21]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c"># instantiate and fit the model to the training data</span>
<span class="kn">from</span> <span class="nn">sklearn.naive_bayes</span> <span class="k">import</span> <span class="n">MultinomialNB</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">MultinomialNB</span><span class="p">()</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">transformed_tweets</span><span class="p">,</span> <span class="n">labels_train</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

To evaluate the model, we predict sentiment on the test set. In order to quantify the performance on the test set, <strong><code>scikit-learn</code></strong>includes a tool to generate a <strong>classification report</strong> from the data.

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [22]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">classification_report</span>

<span class="c"># first vectorize test tweets</span>
<span class="n">vectorized_test_tweets</span> <span class="o">=</span> <span class="n">vectorizer</span><span class="o">.</span><span class="n">transform</span><span class="p">(</span><span class="n">tweets_test</span><span class="p">)</span>
<span class="c"># then transform</span>
<span class="n">transformed_test_tweets</span> <span class="o">=</span> <span class="n">transformer</span><span class="o">.</span><span class="n">transform</span><span class="p">(</span><span class="n">vectorized_test_tweets</span><span class="p">)</span>
<span class="c"># then predict sentiment for the test set</span>
<span class="n">predicted_labels</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">transformed_test_tweets</span><span class="p">)</span> 
<span class="c"># output the results of that prediction in the form of a classfication report</span>
<span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">labels_test</span><span class="p">,</span> <span class="n">predicted_labels</span><span class="p">,</span> <span class="n">target_names</span> <span class="o">=</span> <span class="p">[</span><span class="s">'Negative'</span><span class="p">,</span> <span class="s">'Positive'</span><span class="p">]))</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>             precision    recall  f1-score   support

   Negative       0.75      0.82      0.78    159808
   Positive       0.80      0.72      0.76    160192

avg / total       0.77      0.77      0.77    320000

</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2>3. Determining the Optimal Model</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

Each component of the above process can be configured with optional hyperparameters. To yield the optimal model, it is necessary to perform a grid search on the values of the hyperparamers for every component in the model. Fortunately, <strong><code>scikit-learn</code></strong> offers several tools for rapidly builing and evaluating models with different hyperparameters.

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3>Building a Pipeline</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

We use the <strong><code>Pipeline</code></strong> object to construct a single object that vectorizes, transforms and trains a model on the data.

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [23]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="kn">from</span> <span class="nn">sklearn.pipeline</span> <span class="k">import</span> <span class="n">Pipeline</span>
<span class="c"># create a pipeline object </span>
<span class="n">tweet_classifier</span> <span class="o">=</span> <span class="n">Pipeline</span><span class="p">([(</span><span class="s">'vect'</span><span class="p">,</span> <span class="n">CountVectorizer</span><span class="p">()),</span> 
                             <span class="p">(</span><span class="s">'tfidf'</span><span class="p">,</span> <span class="n">TfidfTransformer</span><span class="p">()),</span> 
                             <span class="p">(</span><span class="s">'clf'</span><span class="p">,</span> <span class="n">MultinomialNB</span><span class="p">())])</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3>Grid Search Parameter Tuning</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

We then use the <strong><code>GridSearchCV</code></strong> tool to search in parameter space for the best performing model, as evaulated by cross-validation on the input data.

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [30]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="kn">from</span> <span class="nn">sklearn.grid_search</span> <span class="k">import</span> <span class="n">GridSearchCV</span>
<span class="kn">import</span> <span class="nn">tokens</span>
<span class="n">parameters</span> <span class="o">=</span> <span class="p">{</span><span class="s">'vect__ngram_range'</span> <span class="p">:</span> <span class="p">[(</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">),</span> <span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">)],</span> <span class="c"># try vectorizing unigrams only or unigrams + bigrams</span>
              <span class="s">'vect__tokenizer'</span> <span class="p">:</span> <span class="p">(</span><span class="n">tokens</span><span class="o">.</span><span class="n">tokenize</span><span class="p">,</span> <span class="k">None</span><span class="p">),</span> <span class="c"># try it with and without Rick's custom tokenizer</span>
              <span class="s">'tfidf__use_idf'</span> <span class="p">:</span> <span class="p">(</span><span class="k">True</span><span class="p">,</span> <span class="k">False</span><span class="p">),</span> <span class="c"># try with and without idf</span>
              <span class="s">'clf__alpha'</span> <span class="p">:</span> <span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="n">e</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="n">e</span><span class="o">-</span><span class="mi">2</span><span class="p">)</span> <span class="p">}</span> <span class="c"># try different values of the alpha parameter</span>

<span class="c"># create a new grid search object using the pipeline and parameters above, which optimizes for f1 score</span>
<span class="n">gridsearch_classifier</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">tweet_classifier</span><span class="p">,</span> <span class="n">parameters</span><span class="p">,</span> <span class="n">scoring</span><span class="o">=</span><span class="s">'f1'</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [31]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c"># run the grid search over the training data</span>
<span class="n">gridsearch_classifier</span> <span class="o">=</span> <span class="n">gridsearch_classifier</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">tweets_train</span><span class="p">,</span> <span class="n">labels_train</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [32]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c"># output the best score achieved in the grid search</span>
<span class="n">gridsearch_classifier</span><span class="o">.</span><span class="n">best_score_</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[32]:</div>
<div class="output_text output_subarea output_execute_result">
<pre>0.79549641365139689</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

The best model resulting from this process is <strong><code>gridsearch_classifier.best_estimator_</code></strong> and can be evaluated on the held-out test set data which it has never seen, and a <strong>classification_report</strong> can be generated.

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [34]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c"># predict sentiment labels using the best estimator from the grid search protocol</span>
<span class="n">predicted_labels</span> <span class="o">=</span> <span class="n">gridsearch_classifier</span><span class="o">.</span><span class="n">best_estimator_</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">tweets_test</span><span class="p">)</span>
<span class="c"># output the classification report</span>
<span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">labels_test</span><span class="p">,</span> <span class="n">predicted_labels</span><span class="p">,</span> <span class="n">target_names</span><span class="o">=</span><span class="p">[</span><span class="s">'Negative'</span><span class="p">,</span> <span class="s">'Positive'</span><span class="p">]))</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>             precision    recall  f1-score   support

   Negative       0.79      0.83      0.81    159808
   Positive       0.82      0.77      0.80    160192

avg / total       0.80      0.80      0.80    320000

</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3>Testing a Linear Support Vector Machine (SVM) Model</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

Using the tools explained above, we can rapidly test a linear SVM model with multiple hyperparameters.

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [36]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="k">import</span> <span class="n">SGDClassifier</span>

<span class="c"># create the pipeline for an SVM classifier</span>
<span class="n">tweet_classifier_SVM</span> <span class="o">=</span> <span class="n">Pipeline</span><span class="p">([(</span><span class="s">'vect'</span><span class="p">,</span> <span class="n">CountVectorizer</span><span class="p">()),</span> 
                             <span class="p">(</span><span class="s">'tfidf'</span><span class="p">,</span> <span class="n">TfidfTransformer</span><span class="p">()),</span> 
                             <span class="p">(</span><span class="s">'clf'</span><span class="p">,</span> <span class="n">SGDClassifier</span><span class="p">(</span><span class="n">loss</span><span class="o">=</span><span class="s">'hinge'</span><span class="p">,</span> 
                                                   <span class="n">penalty</span><span class="o">=</span><span class="s">'l2'</span><span class="p">,</span> 
                                                   <span class="n">n_iter</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> 
                                                   <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">))])</span>
<span class="c"># create a dict of parameters to try </span>
<span class="n">SVM_parameters</span> <span class="o">=</span> <span class="p">{</span><span class="s">'vect__ngram_range'</span> <span class="p">:</span> <span class="p">[(</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">),</span> <span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">)],</span> <span class="c"># try vectorizing unigrams only or unigrams + bigrams</span>
              <span class="s">'vect__tokenizer'</span> <span class="p">:</span> <span class="p">(</span><span class="n">tokens</span><span class="o">.</span><span class="n">tokenize</span><span class="p">,</span> <span class="k">None</span><span class="p">),</span> <span class="c"># try it with and without Rick's custom tokenizer</span>
              <span class="s">'tfidf__use_idf'</span> <span class="p">:</span> <span class="p">(</span><span class="k">True</span><span class="p">,</span> <span class="k">False</span><span class="p">),</span> <span class="c"># try it with and without idf</span>
              <span class="s">'clf__alpha'</span> <span class="p">:</span> <span class="p">(</span><span class="mi">1</span><span class="n">e</span><span class="o">-</span><span class="mi">5</span><span class="p">,</span> <span class="mi">1</span><span class="n">e</span><span class="o">-</span><span class="mi">6</span><span class="p">,</span> <span class="mi">1</span><span class="n">e</span><span class="o">-</span><span class="mi">7</span><span class="p">)</span> <span class="p">}</span> <span class="c"># try different values of the alpha parameter</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [37]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">gridsearch_classifier_SVM</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">tweet_classifier_SVM</span><span class="p">,</span> <span class="n">SVM_parameters</span><span class="p">,</span> <span class="n">scoring</span><span class="o">=</span><span class="s">'f1'</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [39]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">gridsearch_classifier_SVM</span> <span class="o">=</span> <span class="n">gridsearch_classifier_SVM</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">tweets_train</span><span class="p">,</span> <span class="n">labels_train</span><span class="p">)</span> 
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [40]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">gridsearch_classifier_SVM</span><span class="o">.</span><span class="n">best_score_</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[40]:</div>
<div class="output_text output_subarea output_execute_result">
<pre>0.82071665529867655</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

As before, test the <strong><code>best_estimator_</code></strong> of the linear SVM grid search process on the held-out dataset.

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [43]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c"># predict sentiment labels using the best estimator from the grid search protocol</span>
<span class="n">predicted_labels</span> <span class="o">=</span> <span class="n">gridsearch_classifier_SVM</span><span class="o">.</span><span class="n">best_estimator_</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">tweets_test</span><span class="p">)</span>
<span class="c"># output the classification report</span>
<span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">labels_test</span><span class="p">,</span> <span class="n">predicted_labels</span><span class="p">,</span> <span class="n">target_names</span><span class="o">=</span><span class="p">[</span><span class="s">'Negative'</span><span class="p">,</span> <span class="s">'Positive'</span><span class="p">]))</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>             precision    recall  f1-score   support

   Negative       0.81      0.84      0.83    159808
   Positive       0.84      0.81      0.82    160192

avg / total       0.82      0.82      0.82    320000

</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2>4. Model Evalutation on Hand-Labeled Data</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

As a more realistic test of the relative performance of these classifiers, we can benchmark them on a hand-labled dataset of Caltrain-specific tweets. This dataset consists of ~1500 tweets collected over 5 days and labeled as either negative ( 0 ) or positive/neutral ( 1 ) .

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [44]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c"># read in the handlabeled data</span>
<span class="n">benchmark_data</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s">'handlabeled_tweets.csv'</span><span class="p">,</span> <span class="n">index_col</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
<span class="c"># clean up the data</span>
<span class="n">benchmark_data</span><span class="p">[</span><span class="s">'polarity'</span><span class="p">]</span> <span class="o">=</span> <span class="n">benchmark_data</span><span class="p">[</span><span class="s">'sentiment'</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">sent</span><span class="p">:</span> <span class="mi">0</span> <span class="k">if</span> <span class="n">sent</span> <span class="o">==</span> <span class="mi">0</span> <span class="k">else</span> <span class="mi">1</span><span class="p">)</span>
<span class="c"># assign the targets</span>
<span class="n">target_labels</span> <span class="o">=</span> <span class="n">benchmark_data</span><span class="p">[</span><span class="s">'polarity'</span><span class="p">]</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [46]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">best_nb</span> <span class="o">=</span> <span class="n">gridsearch_classifier</span><span class="o">.</span><span class="n">best_estimator_</span> <span class="c"># the best Naive Bayes model</span>
<span class="n">best_svm</span> <span class="o">=</span> <span class="n">gridsearch_classifier_SVM</span><span class="o">.</span><span class="n">best_estimator_</span> <span class="c"># the best SVM model</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3>Naive Bayes Model</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [48]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c"># predict classes using the best NB classifier</span>
<span class="n">nb_pred</span> <span class="o">=</span> <span class="n">best_nb</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">benchmark_data</span><span class="p">[</span><span class="s">'text'</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">target_labels</span><span class="p">,</span> <span class="n">nb_pred</span><span class="p">,</span> <span class="n">target_names</span><span class="o">=</span><span class="p">[</span><span class="s">'Negative'</span><span class="p">,</span> <span class="s">'Positive'</span><span class="p">]))</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>             precision    recall  f1-score   support

   Negative       0.63      0.78      0.70       376
   Positive       0.71      0.55      0.62       376

avg / total       0.67      0.66      0.66       752

</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3>Support Vector Machine Model</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [49]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c"># predict classes using the best SVM classifier</span>
<span class="n">svm_pred</span> <span class="o">=</span> <span class="n">best_svm</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">benchmark_data</span><span class="p">[</span><span class="s">'text'</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">target_labels</span><span class="p">,</span> <span class="n">svm_pred</span><span class="p">,</span> <span class="n">target_names</span><span class="o">=</span><span class="p">[</span><span class="s">'Negative'</span><span class="p">,</span> <span class="s">'Positive'</span><span class="p">]))</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>             precision    recall  f1-score   support

   Negative       0.71      0.74      0.73       376
   Positive       0.73      0.70      0.71       376

avg / total       0.72      0.72      0.72       752

</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

As indicated by its performance on the test set, the best <strong>support vector machine model outperforms the naive bayes model</strong> on this more realistic test set.

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2>The Vocabulary of Positive / Negative Caltrain Tweets</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

Using the SVM model described above, we are able to classify a large group of caltrain-specific tweets by sentiment. Below is a wordcloud visualization of the vocabulary used in positive and negative tweets about caltrain.

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3>Positive</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [86]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="kn">from</span> <span class="nn">IPython.display</span> <span class="k">import</span> <span class="n">Image</span>
<span class="n">Image</span><span class="p">(</span><span class="n">filename</span><span class="o">=</span><span class="s">'sentiment-positive.png'</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[86]:</div>
<div class="output_png output_subarea output_execute_result"></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3>Negative</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [85]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">Image</span><span class="p">(</span><span class="n">filename</span><span class="o">=</span><span class="s">'sentiment-negative.png'</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[85]:</div>
<div class="output_png output_subarea output_execute_result"></div>
</div>
</div>
</div>
</div>