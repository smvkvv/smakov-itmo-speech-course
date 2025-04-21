### Implementation

Unfortunately, I couldn't implement beam search with lm, only lm rescoring. 

### Evaluation

For evaluation I used normalized levenshtein distance from https://www.cse.lehigh.edu/%7Elopresti/Publications/1996/sdair96.pdf

Where 1.0 is the exact match between two strings.

### Results

The LM rescoring didn't show any improvement in metrics, possibly due to the use of CTC approach.

Here is the table in markdown format:

#### 3-gram.pruned.1e-7
| Audio             | Greedy Decoding | Beam Decoding   | Beam_LM_Rescore |
|-------------------|-----------------|-----------------|-----------------|
| examples/sample1.wav | 0.9593965583621628 | 0.9593965583621628 | 0.9593965583621628 |
| examples/sample2.wav | 1.0             | 1.0             | 1.0             |
| examples/sample3.wav | 0.9939209914220377 | 0.9939209914220377 | 0.9939209914220377 |
| examples/sample4.wav | 0.9805831403241088 | 0.9805831403241088 | 0.9805831403241088 |
| examples/sample5.wav | 0.9939940121024164 | 0.9939940121024164 | 0.9939940121024164 |
| examples/sample6.wav | 0.6432791767011973 | 0.6432791767011973 | 0.6432791767011973 |
| examples/sample7.wav | 0.756775583493859  | 0.756775583493859  | 0.756775583493859  |
| examples/sample8.wav | 0.8254877774932579 | 0.8254877774932579 | 0.8254877774932579 |

#### 4-gram.arpa.gz
| Audio             | Greedy Decoding | Beam Decoding   | Beam_LM_Rescore |
|-------------------|-----------------|-----------------|-----------------|
| examples/sample1.wav | 0.9593965583621628 | 0.9593965583621628 | 0.9593965583621628 |
| examples/sample2.wav | 1.0             | 1.0             | 1.0             |
| examples/sample3.wav | 0.9939209914220377 | 0.9939209914220377 | 0.9939209914220377 |
| examples/sample4.wav | 0.9805831403241088 | 0.9805831403241088 | 0.9805831403241088 |
| examples/sample5.wav | 0.9939940121024164 | 0.9939940121024164 | 0.9939940121024164 |
| examples/sample6.wav | 0.6432791767011973 | 0.6432791767011973 | 0.6432791767011973 |
| examples/sample7.wav | 0.756775583493859  | 0.756775583493859  | 0.756775583493859  |
| examples/sample8.wav | 0.8254877774932579 | 0.8254877774932579 | 0.8254877774932579 |
