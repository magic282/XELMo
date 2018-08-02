from ELMo.elmo import Elmo, batch_to_ids

"""
For more detailed how-to, please refer to https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
"""

options_file = r"D:\Users\v-qizhou\Data\ELMo\models\allennlp\elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = r"D:\Users\v-qizhou\Data\ELMo\models\allennlp\elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentences = [['First', 'sentence', '.'], ['Another', '.']]
character_ids = batch_to_ids(sentences)

embeddings = elmo(character_ids)

print(embeddings)

# embeddings['elmo_representations'] is length two list of tensors.
# Each element contains one layer of ELMo representations with shape
# (2, 3, 1024).
#   2    - the batch size
#   3    - the sequence length of the batch
#   1024 - the length of each ELMo vector
