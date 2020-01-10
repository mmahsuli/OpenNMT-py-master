# Author: Mohammad Mahdi Mahsuli
# Show corpus stats of the input files
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--train_src", metavar="FILE")
parser.add_option("--train_tgt", metavar="FILE")
parser.add_option("--valid_src", metavar="FILE")
parser.add_option("--valid_tgt", metavar="FILE")
parser.add_option("--test_src", metavar="FILE")
parser.add_option("--test_tgt", metavar="FILE")

(options, args) = parser.parse_args()

def count_vocab(data_loc): 
	f = open(data_loc, 'r')
	lines = 0
	words = 0
	vocab = []
	while True:
		line = f.readline()
		if not line:
			break;	
		lines = lines + 1	
		for w in line.split():
			words = words + 1
			if w not in vocab:
				vocab.append(w)
	f.close()
	return vocab, lines, words

def count_oov(data_loc, ref_vocab): 
	f = open(data_loc, 'r')
	lines = 0
	words = 0
	oov = []
	while True:
		line = f.readline()
		if not line:
			break;	
		lines = lines + 1	
		for w in line.split():
			words = words + 1
			if w not in ref_vocab and w not in oov:
				oov.append(w)
	f.close()
	return oov, lines, words

train_src_vocab, train_src_lines, train_src_words = count_vocab(options.train_src)
train_tgt_vocab, train_tgt_lines, train_tgt_words = count_vocab(options.train_tgt)

valid_src_oov, valid_src_lines, valid_src_words = count_oov(options.valid_src, train_src_vocab)
valid_tgt_oov, valid_tgt_lines, valid_tgt_words = count_oov(options.valid_tgt, train_tgt_vocab)

test_src_oov, test_src_lines, test_src_words = count_oov(options.test_src, train_src_vocab)
test_tgt_oov, test_tgt_lines, test_tgt_words = count_oov(options.test_tgt, train_tgt_vocab)

print("Train\n==========")
print("Source:")
print("Lines: {0}\tWords: {1}\tVocab-Size: {2}".format(train_src_lines, train_src_words, len(train_src_vocab)))
print("Target:")
print("Lines: {0}\tWords: {1}\tVocab-Size: {2}".format(train_tgt_lines, train_tgt_words, len(train_tgt_vocab)))
print("Valid\n==========")
print("Source:")
print("Lines: {0}\tWords: {1}\tOOV: {2}".format(valid_src_lines, valid_src_words, len(valid_src_oov)))
print("Target:")
print("Lines: {0}\tWords: {1}\tOOV: {2}".format(valid_tgt_lines, valid_tgt_words, len(valid_tgt_oov)))
print("Test\n==========")
print("Source:")
print("Lines: {0}\tWords: {1}\tOOV: {2}".format(test_src_lines, test_src_words, len(test_src_oov)))
print("Target:")
print("Lines: {0}\tWords: {1}\tOOV: {2}".format(test_tgt_lines, test_tgt_words, len(test_tgt_oov)))

