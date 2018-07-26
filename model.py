import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        # Dropout just in case of overfitting (most likely don't need it)
        drop_prob = 0
        
        # embedding layer to convert words to vectors
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # create the lstm
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,
                                dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        self.linear = nn.Linear(hidden_size,vocab_size)
    
    def forward(self, features, captions):
        # Create embedded word vectors for each word in captions
        embeddings = self.embed(captions[:,:-1])
        
        # Stack features and captions
        embeddings = torch.cat((features.unsqueeze(1),embeddings),1)
        
        # Pass lstm over word embeddings to get output and hidden state
        lstm_out, hidden_state = self.lstm(embeddings)
        
        # Feed output through fc layer (and dropout)
        outputs = self.dropout(self.linear(lstm_out))
        
        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output_ids = []
        
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs,states)
            outputs = self.linear(lstm_out)
            _, max_output = outputs.max(2)

            output_ids.append(max_output.item())
            # If we predict <end> then we're done
            if max_output.item() == 1:
                break
            inputs = self.embed(max_output)
            
        return output_ids