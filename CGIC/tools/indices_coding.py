import heapq
import os

"""
Referenced https://github.com/bhrigu123/huffman-coding.
Thanks to Bhrigu Srivastava.
"""

class HuffmanCoding:
	def __init__(self, frequency):
		# self.path = path
		self.heap = []
		self.codes = {}
		self.reverse_mapping = {}
		self.make_heap(frequency)
		self.merge_nodes()
		self.make_codes()

	class HeapNode:
		def __init__(self, char, freq):
			self.char = char
			self.freq = freq
			self.left = None
			self.right = None

		def __lt__(self, other):
			return self.freq < other.freq

		def __eq__(self, other):
			if(other == None):
				return False
			if(not isinstance(other, HeapNode)):
				return False
			return self.freq == other.freq

	# functions for compression:
	def make_frequency_dict(self, text, frequency_codebook):
		frequency = {}
		for character in text:
			if not character in frequency:
				frequency[character] = 0
			# frequency[character] = frequency_codebook[character]
			frequency[character] = int(frequency_codebook[str(character)].item())
		return frequency

	def make_heap(self, frequency):
		for key, value in frequency.items():
			node = self.HeapNode(int(key), int(value.item()))
			heapq.heappush(self.heap, node)

	def merge_nodes(self):
		while(len(self.heap)>1):
			node1 = heapq.heappop(self.heap)
			node2 = heapq.heappop(self.heap)

			merged = self.HeapNode(None, node1.freq + node2.freq)
			merged.left = node1
			merged.right = node2

			heapq.heappush(self.heap, merged)

	def make_codes_helper(self, root):
		stack = [(root, "")]
		while stack:
			node, current_code = stack.pop()
			if node is not None:
				if node.char is not None:
					self.codes[node.char] = current_code
					self.reverse_mapping[current_code] = node.char 
				stack.append((node.right, current_code + "1"))
				stack.append((node.left, current_code + "0"))

	def make_codes(self):
		root = heapq.heappop(self.heap)
		self.make_codes_helper(root)


	def get_encoded_text(self, text):
		encoded_text = ""
		for character in text:
			encoded_text += self.codes[character]
		return encoded_text
	
	def get_encoded_text_bitstudy(self, text):
		encoded_text = []
		for character in text:
			encoded_text.append(self.codes[character])
		return encoded_text


	def pad_encoded_text(self, encoded_text):
		extra_padding = 8 - len(encoded_text) % 8 # extra_padding = (8 - len(encoded_text) % 8) % 8
		for i in range(extra_padding):
			encoded_text += "0"

		padded_info = "{0:08b}".format(extra_padding)
		encoded_text = padded_info + encoded_text
		return encoded_text


	def get_byte_array(self, padded_encoded_text):
		if(len(padded_encoded_text) % 8 != 0):
			print("Encoded text not padded properly")
			exit(0)

		b = bytearray()
		for i in range(0, len(padded_encoded_text), 8):
			byte = padded_encoded_text[i:i+8]
			b.append(int(byte, 2))
		return b


	def compress(self, info, output_path):
		with open(output_path, 'wb') as output:
			text = info.tolist()
			if not text:
				output.write(b'')
				return output_path

			encoded_text = self.get_encoded_text(text)
			padded_encoded_text = self.pad_encoded_text(encoded_text)

			b = self.get_byte_array(padded_encoded_text)
			output.write(bytes(b))

		return output_path


	""" functions for decompression: """

	def remove_padding(self, padded_encoded_text):
		padded_info = padded_encoded_text[:8]
		extra_padding = int(padded_info, 2)

		padded_encoded_text = padded_encoded_text[8:] 
		encoded_text = padded_encoded_text[:-1*extra_padding]

		return encoded_text

	def decode_text(self, encoded_text):
		current_code = ""
		decoded_text = []

		for bit in encoded_text:
			current_code += bit
			if(current_code in self.reverse_mapping):
				character = self.reverse_mapping[current_code]
				decoded_text.append(character)
				current_code = ""

		return decoded_text
	
	def decompress_string(self, path):
		with open(path, 'rb') as file:
			bit_string = ""

			byte = file.read(1)
			if len(byte) == 0:
				return None
			while(len(byte) > 0):
				byte = ord(byte)
				bits = bin(byte)[2:].rjust(8, '0')
				bit_string += bits
				byte = file.read(1)
			encoded_text = self.remove_padding(bit_string)
			decompressed_text = self.decode_text(encoded_text)

		return decompressed_text
