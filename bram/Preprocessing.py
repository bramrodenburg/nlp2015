import os
import xml.sax
import xml.sax.saxutils
from xml.etree import ElementTree as etree
from xml.dom.minidom import parseString
import re
import string
import time
import thread
import threading
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer

DATA_DIRECTORY = "sorted_data/"

class ContentHandler(xml.sax.ContentHandler):
	def __init__(self):
		self.review_text = ""
		self.is_in_review_text = False
		self.punc_tokenizer = RegexpTokenizer(r'\w+')

	def startElement(self, name, attrs):
		if name == "review_text":
			self.is_in_review_text = True

	def endElement(self, name):
		if name == "review_text":
			self.is_in_review_text = False

	def process_words(self, words):
		None 

	def process_data(self, data):
		sentences = sent_tokenize(data)

	def characters(self, data):
		if self.is_in_review_text:
			self.process_data(data)

def process_file(file_name, root):
	file_path = os.path.join(root, file_name)

	print "Started processing %s" % file_path
	file_content = "<root>" + open(file_path, 'r').read() + "</root>"
	
	# Remove special characters
	file_content = re.sub(r'&', '', file_content) # Replace '&' sign for ''
	file_content = filter(lambda x: x in string.printable, file_content) # Remove non printable characters	
	# Remove some HTML tags (very ugly way of doing it)
	file_content = file_content.replace("<br>",'')
	file_content = file_content.replace("\"", '')
	file_content = file_content.replace("<span>", '')
	file_content = file_content.replace("<a>", '')
	file_content = file_content.replace("</span>", '')
	file_content = file_content.replace("</a>", '')
	file_content = file_content.replace("<P>", '')
	file_content = file_content.replace("<The innocent voyage>", '')
	file_content = file_content.replace("<BR>", '')
	file_content = file_content.replace("<ewold@mailexcite.com>", '')
	file_content = file_content.replace("<I>", '')
	file_content = file_content.replace("</I>", '')
	file_content = file_content.replace("<i>", '')
	file_content = file_content.replace("</i>", '')
	file_content = file_content.replace("\x0C", '')
	file_content = file_content.replace("<em>", '')
	file_content = file_content.replace("</em>", '')
	file_content = unicode(file_content)

	xml.sax.parseString(file_content, ContentHandler())

	print "Finished processing file %s" % file_path

def process_files(filenames, root):
	instances = []
	for file_name in filenames:
		if file_name == "all.review":
			instances.append(threading.Thread(target=process_file, args=(file_name, root)))
			instances[-1].start()

def process_data(data_directory):
	for root, dirs, filenames in os.walk(data_directory):
		process_files(filenames, root)

process_data(DATA_DIRECTORY)
