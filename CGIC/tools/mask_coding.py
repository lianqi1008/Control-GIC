import heapq
import os
import zipfile
import numpy as np
import torch


class BinaryCoding:
    def __init__(self):
        self.heap = []
        self.codes = {0: '0', 1: '1'}
        self.reverse_mapping = {'0': 0, '1': 1}

    def get_encoded_text(self, text):
        encoded_text = ""
        for character in text:
            encoded_text += self.codes[character]
        return encoded_text

    def pad_encoded_text(self, encoded_text):
        extra_padding = 8 - len(encoded_text) % 8
        for i in range(extra_padding):
            encoded_text += "0"

        padded_info = "{0:08b}".format(extra_padding)
        encoded_text = padded_info + encoded_text
        return encoded_text

    def get_byte_array(self, padded_encoded_text):
        if (len(padded_encoded_text) % 8 != 0):
            print("Encoded text not padded properly")
            exit(0)

        b = bytearray()
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i + 8]
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
        encoded_text = padded_encoded_text[:-1 * extra_padding]

        return encoded_text

    def decode_text(self, encoded_text):
        current_code = ""
        decoded_text = []

        for bit in encoded_text:
            current_code += bit
            if (current_code in self.reverse_mapping):
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
            while (len(byte) > 0):
                byte = ord(byte)
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
                byte = file.read(1)
        encoded_text = self.remove_padding(bit_string)
        decompressed_text = self.decode_text(encoded_text)

        return decompressed_text