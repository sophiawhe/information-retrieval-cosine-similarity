# Sophia He
# EN.605.744 Information Retrieval
# Program 3
# Fall 2021

# - inverted file structure must be written to disk as a binary file
# - dictionary must be written to disk
# - for each word in the lexicon store a file offset to
#   the corresponding on-disk posting list
# - process the source text file only once

# Notes:
# df = number of documents appears in
# tf = how many times the term appears per document, the number of occurrences of term t in document d
# N = total number of documents
# IDF = Log_2(N)/df
# length = sqrt(tf1^2 + tf2^2 +tf3^2)

import string
import math
import ast
import time
# four bytes stored
SIZE = 4
# tuple stored as (docid, count) pair
PAIR_SIZE = 2
name = "cord19"

def text_normalization(file_name):
    """Perform normalization of the text"""
    with open(file_name, "r") as file:
        FileContent = file.read()
        
    # lower case words
    FileContent = FileContent.lower()

    removal_list = string.punctuation
    removal_list = removal_list.replace('<','')
    removal_list = removal_list.replace('>','')
    # remove punctuation
    x = FileContent.translate(str.maketrans('', '', removal_list))
    # replace newline char with spaces
    x = x.replace('\n', ' ')
    # split on spaces
    x = x.split(' ')
    return x
    
def process_file(file_name):
    """create index from the file"""
    # time the process
    start = time.time()
    # normalize the file
    x = text_normalization(file_name)
    
    # using python dicts
    # {word: [(docid, term_count),(docid, term_count)]}
    vocab = {}
    
    count_paragraphs = 0 # number of paragraphs in file
    count_words = 0 # number of words

    for word in x:
    
        # ignore empty string and p tags
        ignore = ['', '<p>', '<p']
        # if word is a p tag, increment number of paragraphs
        if word == '<p':
            count_paragraphs += 1
            
        # if word not empty and not a p tag
        if word not in ignore and word[len(word)-1] != '>':
            count_words += 1
            
            # if word not seen
            if not vocab.get(word):
                # add to vocab and include current (docid, term_count)
                vocab[word] = [(count_paragraphs, 1)]
            else:
                count_list = vocab[word]
                # if current docid not in list
                if vocab[word][len(vocab[word]) - 1][0] != count_paragraphs:
                    # append (docid, 1) to vocab[word]
                    vocab[word].append((count_paragraphs, 1))
                else:
                    # get tuple from list
                    current_tuple = vocab[word][len(vocab[word]) - 1]
                    # increment count by 1
                    term_count = current_tuple[1] + 1
                    # delete old tuple and replace with new count
                    vocab[word][len(vocab[word]) - 1] = (count_paragraphs, term_count)
    
    # vocab = dict(sorted(vocab.items())) # remove
    inverted = [] # inverted list
    offset_dict = {} # dict {word: (df, offset)}
    print('number of documents:', count_paragraphs)
    print('size of the vocabulary:', len(vocab.keys()))
    print('total terms observed in collection:', count_words)
    
    # set up offset_dict
    offset = 0
    for item in vocab:
        df = 0
        for tup in vocab[item]:
            df += 1 # increment doc freq
            inverted.append(tup[0]) # doc id
            inverted.append(tup[1]) # term count
        offset_dict[item] = (df, offset) # add to dict
        offset += df*PAIR_SIZE # increment offset
    
    # open output files
    f = open(file_name.strip(".txt")+"output.bin", "wb")
    dict_f = open(file_name.strip(".txt")+"dictoutput.txt", "w")
    
    # convert to bin
    byte_array = [num.to_bytes(SIZE, byteorder='big') for num in inverted]
    byte_array = b"".join(byte_array)
    
    # write to output files
    f.write(byte_array)
    dict_f.write(str(offset_dict))
    
    # close output files
    f.close()
    dict_f.close()
    
    # end the time
    end = time.time()
    print("Runtime to build index: ", end - start, "sec")
    return count_paragraphs

def process_query(file_name):
    """Process query information"""
    # normalize the text
    x = text_normalization(file_name)
    
    # tf array of dictionaries
    # [{word: tf}]
    weights = []
    
    count_queries = 0 # number of queries in file
    count_words = 0 # number of words

    for word in x:
        # ignore empty string and q tags
        ignore = ['', '<q>', '<q']
        # if word is a p tag, increment number of paragraphs
        if word == '<q':
            count_queries += 1
            weights.append({})
            
        # if word not empty and not a q tag
        if word not in ignore and word[len(word)-1] != '>':
            count_words += 1
            # if word not in last dict of weights
            if word in weights[len(weights)-1]:
                weights[len(weights)-1][word] += 1
            else:
                weights[len(weights)-1][word] = 1
                
    # print query weights for first query
    print('query weights for first query of ', file_name, weights[0])
    # write to file
    weights_f = open(file_name.strip(".txt")+"queryweightsoutput.txt", "w")
    weights_f.write(str(weights))
    weights_f.close()

def read_from_offset(file, offset):
    """Fetch four bytes from file at offset"""
    file.seek(offset)
    return file.read(SIZE)

def doc_freq_postings(term, file, open_dict):
    """returns (term, freq, postings_list) for a certain term"""
    arr = [] # initialize postings list
    if not open_dict.get(term):
        return (term, 0, [])
    # start at offset
    start = open_dict[term][1]
    # use df to find start of next offset (end of block)
    end = open_dict[term][1]+(PAIR_SIZE*open_dict[term][0])
    # increment by 2
    increment = PAIR_SIZE
    for i in range(start, end, increment):
        id = int.from_bytes(read_from_offset(file, i*SIZE), 'big')
        ct = int.from_bytes(read_from_offset(file, (i+1)*SIZE), 'big')
        arr.append((id,ct))
    freq = open_dict.get(term)[0] # frequency
    return(term, freq, arr)

def open_file_as_dict(filename):
    """process the file as a dictionary or array of dictionaries"""
    file_dict = open(filename,"r")
    # read the file as dict
    open_dict = ast.literal_eval(file_dict.read())
    file_dict.close()
    return open_dict

def IDF(N, df):
    """returns IDF given N and df"""
    # IDF = Log_2(N/df)
    return math.log2(N/df)

def calculate_idf(open_dict, N):
    """
    Create a dictionary of idf per term given index
    open_dict: index dictionary
    N: total number of documents
    """
    # idf_dict: {term: IDF}
    idf_dict = {}
    get_freqs = open_dict.keys()
    # calculate IDF for each term
    for term in get_freqs:
        # document frequency
        df = open_dict.get(term)[0]
        # get IDF
        idf = IDF(N, df)
        # set idf in idf dict
        idf_dict[term] = idf
    return idf_dict

def calculate_lengths(open_dict, file_postings, idf_dict, query_weights, N):
    """
    calculates tfidf, document lengths and query lengths
    open_dict: index dictionary
    file_postings: postings list
    idf_dict: dictionary of idf per term
    query_weights: list of dictionaries of weights for each query
    N: total number of documents
    """
    # tfidfs calculated for each document
    doc_tfidf = [{} for i in range(N)]
    # initialize array of lengths for each document
    lengths = [0]*(N)
    # get frequencies and postings
    for term in open_dict.keys():
        # get terms, frequencies, and array of (docID, tf) pairs
        terms, freqs, arr = doc_freq_postings(term, file_postings, open_dict)
        for pair in arr:
            # sum square of tfidf for lengths
            lengths[pair[0]-1] += (idf_dict[term]*pair[1])**2
            # set tfidf in dictionary
            doc_tfidf[pair[0]-1][term] = idf_dict[term]*pair[1]
    
    # initialize array of lengths for queries
    query_lengths = [0]*len(query_weights)
    for i in range(len(query_weights)):
        for k in query_weights[i].keys():
            # calculate and sum tfidf squares for queries
            tfidf = 0
            if idf_dict.get(k):
                tfidf = query_weights[i][k]*idf_dict[k]
            query_lengths[i] += (tfidf)**2
    
    # square root of sum of squares for each document and query
    lengths = [math.sqrt(i) for i in lengths]
    query_lengths = [math.sqrt(i) for i in query_lengths]
    return doc_tfidf, lengths, query_lengths

def cos_similarities(query_weights, idf_dict, query_lengths, lengths, doc_tfidf, N):
    """
    calculate document similarity scores
    query_weights: list of dictionaries of weights for each query
    idf_dict: dictionary of idf per term
    query_lengths: array of lengths for queries
    lengths: array of lengths for each document
    doc_tfidf: tfidfs calculated for each document
    N: total number of documents
    """
    # list of arrays of cosine similarities per query
    query_cos = []
    # list of dictionaries, dictionary per document stores tfidf
    query_tfidf = [] #TODO may not need this
    # for each query
    for i in range(len(query_weights)):
        # empty dictionary for tfidf array
        query_tfidf.append({})
        # initialize cosine similarity array
        query_cos.append([0]*N)
        # for each nonrepeating word in the query
        for k in query_weights[i].keys():
            # calculate tfidf = weight*idf
            tfidf = 0
            if idf_dict.get(k):
                tfidf = query_weights[i][k]*idf_dict[k]
            # if query
            if(query_lengths[i] != 0):
                query_tfidf[i][k] = tfidf/query_lengths[i]
            # N total cosine similarities calculated per query
            for j in range(N):
                # avoid divide by zero and nonetype errors
                if(query_lengths[i] != 0) & (lengths[j] != 0):
                    if(doc_tfidf[j].get(k)):
                        # add normalized dot product component
                        query_cos[i][j] += (doc_tfidf[j].get(k)/lengths[j])*(tfidf/query_lengths[i])
    return query_cos


def rank_list(query_cos, file_name, jhed, N):
    """
    Output rank list for top 100 documents
    format: queryID Q0 docID rank numericalScore jhed
    
    query_cos: list of arrays of cosine similarities per query
    file_name: name of output file
    jhed: your jhed
    N: total number of documents
    """
    # initialize output array
    output_results = []
    # for each query's cosine similarity array
    for cos_array in query_cos:
        # create output result array of tuples (cosine similarity, query number)
        output_result = [(cos_array[i], i) for i in range(N)]
        # reverse sort
        output_result.sort(reverse=True)
        # append to output array
        output_results.append(output_result)

    # open output file
    output_f = open(file_name, "w")
        
    # write to output file
    for queryID in range(len(output_results)):
        # for the top 100
        for i in range(100):
            # get score and docID from output results
            score, docID = output_results[queryID][i]
            # write line to file
            output_f.write(str(queryID+1)+ " Q0 "+ str(docID))
            output_f.write(" " + str(i+1) + " " + str(score) + " "+jhed)
            output_f.write('\n')
        
    # close output file
    output_f.close()

# Build Index
N = process_file(name + ".txt")

##################################################
# Part A: Topics Query Processing
# Open Index Files
start = time.time() # time
open_dict = open_file_as_dict(name + "dictoutput.txt")
file_postings = open(name + "output.bin","rb")

# Calculate IDF
idf_dict = calculate_idf(open_dict, N)

# process query
process_query(name + ".topics.keyword.txt")
# call functions for calculating cosine similarities
query_weights = open_file_as_dict(name + ".topics.keywordqueryweightsoutput.txt")
doc_tfidf, lengths, query_lengths = calculate_lengths(open_dict, file_postings, idf_dict, query_weights, N)
query_cos = cos_similarities(query_weights, idf_dict, query_lengths, lengths, doc_tfidf, N)
# output rank list
rank_list(query_cos, "she37-a.txt", "she37", N)
end = time.time() # end time
print("Runtime to process topics query: ", end - start, "sec")

##################################################
# Part B: Length Experiment/Questions Query Parsing
# Open Index Files
start = time.time() # time
open_dict = open_file_as_dict(name + "dictoutput.txt")
file_postings = open(name + "output.bin","rb")

# Calculate IDF
idf_dict = calculate_idf(open_dict, N)

# process query
process_query(name + ".topics.question.txt")
# call functions for calculating cosine similarities
query_weights_b = open_file_as_dict(name + ".topics.questionqueryweightsoutput.txt")
doc_tfidf_b, lengths_b, query_lengths_b = calculate_lengths(open_dict, file_postings, idf_dict, query_weights_b, N)
query_cos_b = cos_similarities(query_weights_b, idf_dict, query_lengths_b, lengths_b, doc_tfidf_b, N)
# output rank list
rank_list(query_cos_b, "she37-b.txt", "she37", N)
end = time.time() # end time
print("Runtime to process questions query: ", end - start, "sec")


