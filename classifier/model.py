import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss

from transformers import BertModel, BertPreTrainedModel

class MyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.log_softmax = nn.LogSoftmax()
       
        self.criterion = nn.KLDivLoss(size_average=False)
        self.label_smoothing = 0.0
        self.confidence = 1.0 - self.label_smoothing

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        log_softmax = self.log_softmax(logits)

        loss = None
        if labels is not None:
            with torch.no_grad():
                true_dist = torch.mul(labels, self.confidence)
                true_dist = torch.add(true_dist, self.label_smoothing / (self.num_labels - 1))

            loss = self.criterion(log_softmax, true_dist)


        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class BertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()  # for multi labels
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())

            #         loss_fct = CrossEntropyLoss()  # for single labels
            #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            #         return loss

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

class MyBertForDomainSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(MyBertForDomainSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.log_softmax = nn.LogSoftmax()

        self.criterion = CrossEntropyLoss()
        self.label_smoothing = 0.0
        self.confidence = 1.0 - self.label_smoothing

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # for single labels
            loss = self.criterion(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), dim=1))

        # log_softmax = self.log_softmax(logits)
        #
        # loss = None
        # if labels is not None:
        #     with torch.no_grad():
        #         true_dist = torch.mul(labels, self.confidence)
        #         true_dist = torch.add(true_dist, self.label_smoothing / (self.num_labels - 1))
        #     print("log_softmax", log_softmax)
        #     print("true_dist", true_dist)
        #     print("true_dist.long()", true_dist.long())
        #     loss = self.criterion(log_softmax, true_dist.long())

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class MyBertForSequenceClassificationWithPos(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + 1, config.num_labels)
        self.log_softmax = nn.LogSoftmax()
       
        self.criterion = nn.KLDivLoss(size_average=False)
        self.label_smoothing = 0.0
        self.confidence = 1.0 - self.label_smoothing

        self.init_weights()

    def forward(
        self,
        input_ids=None,        
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,        
        pos_in_doc=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]
        

        classifier_input = torch.cat((pooled_output, pos_in_doc), dim=1)
        #classifier_input = pooled_output
        classifier_input = self.dropout(classifier_input)

        logits = self.classifier(classifier_input)
        log_softmax = self.log_softmax(logits)

        loss = None
        if labels is not None:
            with torch.no_grad():
                true_dist = torch.mul(labels, self.confidence)
                true_dist = torch.add(true_dist, self.label_smoothing / (self.num_labels - 1))

            loss = self.criterion(log_softmax, true_dist)


        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output