---
ID: 38
post_title: >
  Purchase Sequences and Transition
  Matrices
author: Chloe Mawer
post_date: 2016-01-13 19:55:48
post_excerpt: ""
layout: post
permalink: >
  http://techvault.svds.io/wordpress/purchase-sequences-and-transition-matrices/
published: true
vortex_system_likes:
  - "0"
vortex_system_dislikes:
  - "0"
vortex_system_user_1:
  - 'a:2:{s:5:"liked";s:5:"liked";s:8:"disliked";s:8:"disliked";}'
"vortex_system_user_::1":
  - 'a:2:{s:5:"liked";s:5:"liked";s:8:"disliked";s:8:"disliked";}'
---
A number of functions that take a purchase history, labels a customer's history by the mix of items' categories, looks at most common sequences of purchases made according to those labels, and creates transition matrices to percent of people making purchase type A who made purchase type B. Creates a number of figures.

<hr />

<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [1]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="o">%</span><span class="k">matplotlib</span> inline
<span class="kn">import</span> <span class="nn">numpy</span> <span class="kn">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="kn">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">random</span>
<span class="kn">from</span> <span class="nn">matplotlib</span> <span class="kn">import</span> <span class="n">pyplot</span> <span class="k">as</span> <span class="n">plt</span>
<span class="kn">import</span> <span class="nn">sys</span>
<span class="kn">import</span> <span class="nn">time</span>
<span class="kn">import</span> <span class="nn">json</span>
<span class="kn">import</span> <span class="nn">datetime</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="kn">as</span> <span class="nn">sns</span>
<span class="kn">import</span> <span class="nn">matplotlib</span> <span class="kn">as</span> <span class="nn">mpl</span>
<span class="n">mpl_update</span> <span class="o">=</span> <span class="p">{</span><span class="s">'font.size'</span><span class="p">:</span><span class="mi">16</span><span class="p">,</span><span class="s">'xtick.labelsize'</span><span class="p">:</span><span class="mi">14</span><span class="p">,</span><span class="s">'ytick.labelsize'</span><span class="p">:</span><span class="mi">14</span><span class="p">,</span><span class="s">'figure.figsize'</span><span class="p">:[</span><span class="mf">12.0</span><span class="p">,</span><span class="mf">8.0</span><span class="p">],</span><span class="s">'axes.color_cycle'</span><span class="p">:[</span><span class="s">'#0055A7'</span><span class="p">,</span> <span class="s">'#2C3E4F'</span><span class="p">,</span> <span class="s">'#26C5ED'</span><span class="p">,</span> <span class="s">'#00cc66'</span><span class="p">,</span> <span class="s">'#D34100'</span><span class="p">,</span> <span class="s">'#ECEFF0'</span><span class="p">,</span><span class="s">'#FF9700'</span><span class="p">,</span> 
                                 <span class="s">'#091D32'</span><span class="p">],</span> <span class="s">'axes.labelsize'</span><span class="p">:</span><span class="mi">20</span><span class="p">,</span><span class="s">'axes.labelcolor'</span><span class="p">:</span><span class="s">'#677385'</span><span class="p">,</span><span class="s">'axes.titlesize'</span><span class="p">:</span><span class="mi">20</span><span class="p">,</span><span class="s">'lines.color'</span><span class="p">:</span><span class="s">'#0055A7'</span><span class="p">,</span><span class="s">'lines.linewidth'</span><span class="p">:</span><span class="mi">3</span><span class="p">,</span><span class="s">'text.color'</span><span class="p">:</span><span class="s">'#677385'</span><span class="p">}</span>
<span class="n">mpl</span><span class="o">.</span><span class="n">rcParams</span><span class="o">.</span><span class="n">update</span><span class="p">(</span><span class="n">mpl_update</span><span class="p">)</span>
<span class="n">sys</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="s">'/home/cmawer/ds-analytics/SVDS/engagement/womens_seq/src'</span><span class="p">)</span>
<span class="kn">import</span> <span class="nn">sequences</span> <span class="kn">as</span> <span class="nn">seq</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [2]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="o">%</span><span class="k">load_ext</span> autoreload
<span class="o">%</span><span class="k">autoreload</span> 2
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
<h1>Sequential purchase analysis</h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

This notebook shows how to use the functions in sequences.py. It first creates a fake dataset of items ordered by customer and by order (or basket or purchase). It then shows how to assess the most common sequences of purchases where a purchase is described the mix of labels the items within the purchase consist of (such as shoe+sock). Lastly, it makes transition matrices between first and second purchases to show the percent of people with mix_a in their first purchase make a second purchase with mix_b.

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1>Create fake data</h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2>Set parameters of fake data</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [3]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="c"># Number of unique customers</span>
<span class="n">n_customers</span> <span class="o">=</span> <span class="mi">100000</span>

<span class="c"># Date of first order</span>
<span class="n">start_date</span> <span class="o">=</span> <span class="s">'2014-01-01'</span>

<span class="c"># Date of last order considered</span>
<span class="n">end_date</span> <span class="o">=</span> <span class="s">'2014-12-31'</span>

<span class="c"># The percent of people who made an nth order that will make an n+1th order. </span>
<span class="n">fraction_reorder</span> <span class="o">=</span> <span class="mf">0.3</span>

<span class="c"># The percent of people to buy a second item. Half that number will buy a 3rd.</span>
<span class="c"># One third that number will buy a fourth, etc</span>
<span class="n">fraction_multi_items</span> <span class="o">=</span> <span class="mf">0.4</span> 

<span class="c"># The maximum number of orders for a member</span>
<span class="n">max_num_orders</span> <span class="o">=</span> <span class="mi">4</span> 

<span class="c"># The maximum number of items in someone's order</span>
<span class="n">max_num_items</span> <span class="o">=</span> <span class="mi">5</span>

<span class="c"># Possible categories with occurrences representing relative proportion of those category items.</span>
<span class="c"># cats = ['cat1','cat1','cat1','cat2','cat2','cat3'] means 1/2 of items are cat1, 1/3 items cat2 and 1/6 cat3</span>
<span class="n">cats</span> <span class="o">=</span> <span class="p">[</span><span class="s">'cat1'</span><span class="p">,</span><span class="s">'cat1'</span><span class="p">,</span><span class="s">'cat1'</span><span class="p">,</span><span class="s">'cat2'</span><span class="p">,</span><span class="s">'cat2'</span><span class="p">,</span><span class="s">'cat3'</span><span class="p">]</span>
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
<h2>Create customer ids</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [4]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">customer_id</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="n">n_customers</span><span class="p">)</span>
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
<h2>Create dates of first order</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [5]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">num_days</span> <span class="o">=</span> <span class="nb">int</span><span class="p">((</span><span class="n">np</span><span class="o">.</span><span class="n">datetime64</span><span class="p">(</span><span class="n">end_date</span><span class="p">)</span><span class="o">-</span><span class="n">np</span><span class="o">.</span><span class="n">datetime64</span><span class="p">(</span><span class="n">start_date</span><span class="p">))</span><span class="o">/</span><span class="n">np</span><span class="o">.</span><span class="n">timedelta64</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="s">'D'</span><span class="p">))</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [6]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">dates</span> <span class="o">=</span> <span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">datetime64</span><span class="p">(</span><span class="n">start_date</span><span class="p">)</span><span class="o">+</span><span class="n">np</span><span class="o">.</span><span class="n">timedelta64</span><span class="p">(</span><span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="n">num_days</span><span class="p">),</span><span class="s">'D'</span><span class="p">)</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n_customers</span><span class="p">)]</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [7]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">order_num</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">ones</span><span class="p">(</span><span class="n">customer_id</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [8]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">items</span> <span class="o">=</span> <span class="nb">zip</span><span class="p">(</span><span class="n">customer_id</span><span class="p">,</span> <span class="n">dates</span><span class="p">)</span>
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
<h2>Get ids and dates of second, third, fourth purchases</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [11]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="k">def</span> <span class="nf">max_days</span><span class="p">(</span><span class="n">date</span><span class="p">,</span><span class="n">end_date</span><span class="p">):</span>
    <span class="k">return</span> <span class="nb">int</span><span class="p">((</span><span class="n">np</span><span class="o">.</span><span class="n">datetime64</span><span class="p">(</span><span class="n">end_date</span><span class="p">)</span><span class="o">-</span><span class="n">date</span><span class="p">)</span><span class="o">/</span><span class="n">np</span><span class="o">.</span><span class="n">timedelta64</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="s">'D'</span><span class="p">))</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [12]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">orders</span> <span class="o">=</span> <span class="p">[</span><span class="nb">zip</span><span class="p">(</span><span class="n">customer_id</span><span class="p">,</span><span class="n">dates</span><span class="p">)]</span>
<span class="k">for</span> <span class="n">j</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="n">max_num_orders</span><span class="p">):</span>
    <span class="n">next_order</span> <span class="o">=</span> <span class="n">random</span><span class="o">.</span><span class="n">sample</span><span class="p">(</span><span class="n">orders</span><span class="p">[</span><span class="n">j</span><span class="o">-</span><span class="mi">1</span><span class="p">],</span> <span class="nb">int</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="n">fraction_reorder</span><span class="o">*</span><span class="nb">len</span><span class="p">(</span><span class="n">orders</span><span class="p">[</span><span class="n">j</span><span class="o">-</span><span class="mi">1</span><span class="p">]))))</span>
    <span class="n">cust</span><span class="p">,</span> <span class="n">orig_date</span> <span class="o">=</span> <span class="nb">zip</span><span class="p">(</span><span class="o">*</span><span class="n">next_order</span><span class="p">)</span>
    <span class="n">next_date</span> <span class="o">=</span> <span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">timedelta64</span><span class="p">(</span><span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="n">max_days</span><span class="p">(</span><span class="n">x</span><span class="p">,</span><span class="n">end_date</span><span class="p">)),</span><span class="s">'D'</span><span class="p">)</span><span class="o">+</span> <span class="n">x</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">orig_date</span><span class="p">]</span>
    <span class="n">orders</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">cust</span><span class="p">,</span><span class="n">next_date</span><span class="p">))</span>
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
<h2>Make multi-item orders</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [13]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">final_orders</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">for</span> <span class="n">k</span><span class="p">,</span> <span class="n">group</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">orders</span><span class="p">):</span>
    <span class="k">for</span> <span class="n">j</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="n">max_num_items</span><span class="p">):</span>
        <span class="n">group</span> <span class="o">=</span> <span class="n">group</span> <span class="o">+</span> <span class="n">random</span><span class="o">.</span><span class="n">sample</span><span class="p">(</span><span class="n">group</span><span class="p">,</span><span class="nb">int</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="n">fraction_multi_items</span><span class="o">*</span><span class="nb">len</span><span class="p">(</span><span class="n">group</span><span class="p">)</span><span class="o">/</span><span class="n">j</span><span class="p">)))</span>
    <span class="n">final_orders</span><span class="o">=</span> <span class="n">final_orders</span> <span class="o">+</span> <span class="nb">zip</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="o">*</span><span class="n">group</span><span class="p">)[</span><span class="mi">0</span><span class="p">],</span><span class="nb">zip</span><span class="p">(</span><span class="o">*</span><span class="n">group</span><span class="p">)[</span><span class="mi">1</span><span class="p">],((</span><span class="n">k</span><span class="o">+</span><span class="mi">1</span><span class="p">)</span><span class="o">*</span><span class="n">np</span><span class="o">.</span><span class="n">ones</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">group</span><span class="p">))))</span>
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
<h2>Assign categories to each item purchased</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [15]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">categories</span> <span class="o">=</span> <span class="p">[</span><span class="n">random</span><span class="o">.</span><span class="n">choice</span><span class="p">(</span><span class="n">cats</span><span class="p">)</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">final_orders</span><span class="p">))]</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [16]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">purchase_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">final_orders</span><span class="p">)</span>
<span class="n">purchase_df</span><span class="o">.</span><span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s">'customer_id'</span><span class="p">,</span><span class="s">'order_date'</span><span class="p">,</span><span class="s">'order_num'</span><span class="p">]</span>
<span class="n">purchase_df</span><span class="p">[</span><span class="s">'category'</span><span class="p">]</span> <span class="o">=</span> <span class="n">categories</span>
<span class="n">purchase_df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
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
<th>customer_id</th>
<th>order_date</th>
<th>order_num</th>
<th>category</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>1</td>
<td>2014-12-30</td>
<td>1</td>
<td>cat1</td>
</tr>
<tr>
<th>1</th>
<td>2</td>
<td>2014-07-04</td>
<td>1</td>
<td>cat1</td>
</tr>
<tr>
<th>2</th>
<td>3</td>
<td>2014-07-18</td>
<td>1</td>
<td>cat2</td>
</tr>
<tr>
<th>3</th>
<td>4</td>
<td>2014-02-28</td>
<td>1</td>
<td>cat1</td>
</tr>
<tr>
<th>4</th>
<td>5</td>
<td>2014-09-15</td>
<td>1</td>
<td>cat2</td>
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
<h2>Get total number of orders per category</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [17]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">total_purch_num</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">purchase_df</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="s">'customer_id'</span><span class="p">)[</span><span class="s">'order_num'</span><span class="p">]</span><span class="o">.</span><span class="n">nunique</span><span class="p">())</span>
<span class="n">total_purch_num</span><span class="o">.</span><span class="n">columns</span><span class="o">=</span><span class="p">[</span><span class="s">'total_order_num'</span><span class="p">]</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [18]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">purchase_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">merge</span><span class="p">(</span><span class="n">purchase_df</span><span class="p">,</span> <span class="n">total_purch_num</span><span class="o">.</span><span class="n">reset_index</span><span class="p">(),</span> <span class="n">on</span><span class="o">=</span><span class="s">'customer_id'</span><span class="p">,</span> <span class="n">how</span><span class="o">=</span><span class="s">'left'</span><span class="p">)</span>
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
<h2>Final fake dataset</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [19]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">purchase_df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[19]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<table class="dataframe" border="1">
<thead>
<tr>
<th></th>
<th>customer_id</th>
<th>order_date</th>
<th>order_num</th>
<th>category</th>
<th>total_order_num</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>1</td>
<td>2014-12-30</td>
<td>1</td>
<td>cat1</td>
<td>2</td>
</tr>
<tr>
<th>1</th>
<td>2</td>
<td>2014-07-04</td>
<td>1</td>
<td>cat1</td>
<td>2</td>
</tr>
<tr>
<th>2</th>
<td>3</td>
<td>2014-07-18</td>
<td>1</td>
<td>cat2</td>
<td>1</td>
</tr>
<tr>
<th>3</th>
<td>4</td>
<td>2014-02-28</td>
<td>1</td>
<td>cat1</td>
<td>1</td>
</tr>
<tr>
<th>4</th>
<td>5</td>
<td>2014-09-15</td>
<td>1</td>
<td>cat2</td>
<td>2</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [20]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">purchase_df</span><span class="o">.</span><span class="n">total_order_num</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[20]:</div>
<div class="output_text output_subarea output_execute_result">
<pre>1    146831
2     87516
3     39623
4     22806
dtype: int64</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1>Get sequences of purchases labeled by category</h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

<code>transitions.create_sequences()</code> creates sequences of purchases for each customer.

Sequences look like this:

<code>purchase1-purchase2-...-purchaseN</code>

where each purchase looks like:

<code>label1+label2+...+labelN</code>

where labels are those seen in a given basket for a customer.

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [21]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">mix</span><span class="p">,</span> <span class="n">series_purchases</span><span class="p">,</span> <span class="n">sequences</span><span class="p">,</span> <span class="n">string_sequences</span><span class="p">,</span> <span class="n">fig_bar</span> <span class="o">=</span> <span class="n">seq</span><span class="o">.</span><span class="n">create_sequences</span><span class="p">(</span><span class="n">purchase_df</span><span class="p">,</span> <span class="s">'category'</span><span class="p">,</span><span class="s">'Category'</span><span class="p">,</span> <span class="n">identifier</span><span class="o">=</span><span class="s">'customer_id'</span><span class="p">,</span> <span class="n">purch_order_field</span><span class="o">=</span><span class="s">'order_num'</span><span class="p">,</span><span class="n">extra_fields</span><span class="o">=</span><span class="s">'total_order_num'</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [22]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">series_purchases</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[22]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<table class="dataframe" border="1">
<thead>
<tr>
<th></th>
<th>order_num</th>
<th>total_order_num</th>
<th>purchase1</th>
<th>purchase2</th>
<th>purchase3</th>
<th>purchase4</th>
<th>purchase5</th>
<th>purchase6</th>
<th>sequences</th>
<th>string_sequences</th>
</tr>
<tr>
<th>customer_id</th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<th>5248</th>
<td>1</td>
<td>1</td>
<td>cat1+cat2</td>
<td>none</td>
<td>none</td>
<td>none</td>
<td>none</td>
<td>none</td>
<td>[cat1+cat2, none, none, none, none, none]</td>
<td>cat1+cat2-none-none-none-none-none</td>
</tr>
<tr>
<th>53518</th>
<td>1</td>
<td>2</td>
<td>cat1+cat2</td>
<td>cat1+cat2</td>
<td>none</td>
<td>none</td>
<td>none</td>
<td>none</td>
<td>[cat1+cat2, cat1+cat2, none, none, none, none]</td>
<td>cat1+cat2-cat1+cat2-none-none-none-none</td>
</tr>
<tr>
<th>28429</th>
<td>1</td>
<td>1</td>
<td>cat2</td>
<td>none</td>
<td>none</td>
<td>none</td>
<td>none</td>
<td>none</td>
<td>[cat2, none, none, none, none, none]</td>
<td>cat2-none-none-none-none-none</td>
</tr>
<tr>
<th>67806</th>
<td>1</td>
<td>1</td>
<td>cat1+cat3</td>
<td>none</td>
<td>none</td>
<td>none</td>
<td>none</td>
<td>none</td>
<td>[cat1+cat3, none, none, none, none, none]</td>
<td>cat1+cat3-none-none-none-none-none</td>
</tr>
<tr>
<th>20065</th>
<td>1</td>
<td>3</td>
<td>cat1</td>
<td>cat2</td>
<td>cat1</td>
<td>none</td>
<td>none</td>
<td>none</td>
<td>[cat1, cat2, cat1, none, none, none]</td>
<td>cat1-cat2-cat1-none-none-none</td>
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
<h1>Plot top sequences</h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

You can either plot the top sequences for all people who purchased n times:

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [23]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">three_top</span><span class="p">,</span> <span class="n">fig_three</span><span class="o">=</span> <span class="n">seq</span><span class="o">.</span><span class="n">top_sequences</span><span class="p">(</span><span class="n">series_purchases</span><span class="p">,</span> <span class="n">num_purchases</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span> <span class="n">purch_order_field</span><span class="o">=</span><span class="s">'order_num'</span><span class="p">,</span><span class="n">total_order_field</span><span class="o">=</span><span class="s">'total_order_num'</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

Or you can look at the n length sequences for all purchasers (but some will have no orders after the first and others will have had their sequences cut off.

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [24]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">three_top</span><span class="p">,</span> <span class="n">fig_three</span><span class="o">=</span> <span class="n">seq</span><span class="o">.</span><span class="n">top_sequences</span><span class="p">(</span><span class="n">series_purchases</span><span class="p">,</span> <span class="n">num_purchases</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span> <span class="n">limit_to_n_purchasers</span><span class="o">=</span><span class="bp">False</span><span class="p">,</span><span class="n">purch_order_field</span><span class="o">=</span><span class="s">'order_num'</span><span class="p">,</span><span class="n">total_order_field</span><span class="o">=</span><span class="s">'total_order_num'</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1>Transitions</h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [25]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">mix</span><span class="p">,</span> <span class="n">series_purchases</span><span class="p">,</span> <span class="n">trans</span><span class="p">,</span> <span class="n">trans_multi</span><span class="p">,</span> <span class="n">trans_abs</span><span class="p">,</span> <span class="n">fig</span><span class="p">,</span> <span class="n">fig_multi</span><span class="p">,</span> <span class="n">fig_bar</span> <span class="o">=</span> <span class="n">seq</span><span class="o">.</span><span class="n">create_transitions</span><span class="p">(</span><span class="n">purchase_df</span><span class="p">,</span> <span class="s">'category'</span><span class="p">,</span><span class="s">'Category'</span><span class="p">,</span>  <span class="n">identifier</span><span class="o">=</span><span class="s">'customer_id'</span><span class="p">,</span> <span class="n">purch_order_field</span><span class="o">=</span><span class="s">'order_num'</span><span class="p">,</span><span class="n">extra_fields</span><span class="o">=</span><span class="s">'total_order_num'</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "></div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "></div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

Obviously the data above is fake!

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1>Limiting the categories</h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

Sometimes there will be a label or labels that only account for a very small percent of items sold but if included would create a large number of combinations of basket labels so you may want to exclude them. The <code>include</code> keyword argument allow you to say which labels you'd like to include for both sequences and transitions.

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [26]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython2">
<pre><span class="n">mix</span><span class="p">,</span> <span class="n">series_purchases</span><span class="p">,</span> <span class="n">trans</span><span class="p">,</span> <span class="n">trans_multi</span><span class="p">,</span> <span class="n">trans_abs</span><span class="p">,</span> <span class="n">fig</span><span class="p">,</span> <span class="n">fig_multi</span><span class="p">,</span> <span class="n">fig_bar</span> <span class="o">=</span> <span class="n">seq</span><span class="o">.</span><span class="n">create_transitions</span><span class="p">(</span><span class="n">purchase_df</span><span class="p">,</span> <span class="s">'category'</span><span class="p">,</span><span class="s">'Category'</span><span class="p">,</span>  <span class="n">include</span> <span class="o">=</span> <span class="p">[</span><span class="s">'cat1'</span><span class="p">,</span><span class="s">'cat2'</span><span class="p">],</span> <span class="n">identifier</span><span class="o">=</span><span class="s">'customer_id'</span><span class="p">,</span> <span class="n">purch_order_field</span><span class="o">=</span><span class="s">'order_num'</span><span class="p">,</span><span class="n">extra_fields</span><span class="o">=</span><span class="s">'total_order_num'</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "></div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "></div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

The labels not included will appear as blank if there were purchases with only that label. Combination baskets just won't mention them so the <code>cat1+cat2</code> basket above includes also baskets with cat1, cat2, and cat3. The <code>cat1</code> basket will also include baskets with cat1 and cat3.

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell"></div>
</div>
</div>