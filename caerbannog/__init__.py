# -*- coding: utf-8 -*-
import wabbit_wappa as _wabbit_wappa
from math import exp


class Example(object):
    """
    Wraps an example for Vowpal Wabbit.
    """

    def __init__(self,
                 label=None,
                 importance=None,
                 base=None,
                 tag=None,
                 features=None,
                 namespaces=None):

        self.label = label
        self.importance = importance
        self.base = base
        self.tag = tag
        self.namespaces = {}

        if namespaces:
            self.add_namespaces(namespaces)

        if features:
            self.add_namespace(NameSpace(features=features))

    def namespace(self, name=None):
        if not name in namespaces:
            self.add_namespace(Namespace(name=name))

    def add_feature(self, feature, value=None, namespace=None):
        self.namespace(namespace).add_feature(feature, value)

    def add_features(self, features, namespace=None):
        self.namespace(namespace).add_features(features)

    def add_namespaces(self, namespaces):
        for i in namespaces:
            self.add_namespace(i)

    def add_namespace(self, namespace):
        self.namespaces[namespace.name] = namespace

    def make_line(self, label=None, importance=None, base=None, tag=None):
        """Makes and returns an example string in VW syntax.
        If given, 'response', 'importance', 'base', and 'tag' are used
        to label the example.  Features for the example come from
        any given features or namespaces, as well as any previously
        added namespaces (using them up in the process).
        """

        response   = self.label      if label      is None else label
        importance = self.importance if importance is None else importance
        tag        = self.tag        if tag        is None else tag
        base       = self.base       if base       is None else base
 
        substrings = []
        tokens = []
        if response is not None:
            token = str(response)
            tokens.append(token)

            if importance is not None:  # Check only if response is given
                token = str(importance)
                tokens.append(token)
                if base is not None:  # Check only if importance is given
                    token = str(base)
                    tokens.append(token)

        if tag is not None: 
            token = "'" + str(tag)  # Tags are unambiguous if given a ' prefix
            tokens.append(token)

        else:
            token = ""  # Spacing element to avoid ambiguity in parsing
            tokens.append(token)

        substring = ' '.join(tokens)
        substrings.append(substring)

        if self.namespaces:
            for namespace in self.namespaces.values():
                substring = namespace.to_string()
                substrings.append(substring)
        else:
            substrings.append('')  # For correct syntax

        line = '|'.join(substrings)
        return line
    

class LogisticPrediction(object):
    def __init__(self, result):
        self.value = result.value
        if hasattr(result, 'importance'):
            self.importance = result.importance

    @property
    def logistic():
        """
        Returns the 0..1 probability of label being 1
        """

        return 1.0 / (1.0 + exp(-self.value))

    @property
    def logistic_11():
        """
        Returns the value of -1..1 logistic function at
        the resulting value.
        """

        return 1.0 / (2.0 + exp(-self.value)) - 1.0


class Rabbit(object):
    _prediction_factory = LogisticPrediction

    def __init__(self, **kwargs):
        self.options = kwargs

    def start(self):
        self.vw = VW(**self.options)

    def send_line(self, line):
        self.vw.send_line(line)

    def make_line(self, *, example=None, label=None, importance=None, base=None,
                  tag=None, features=None, namespaces=None, no_label=False):

        example = copy.copy(example) or Example()
        if namespaces:
            example.add_namespaces(namespaces)

        if features:
            example.add_features(features)

        if no_label:
            example.label = None

        return example.make_line(label=label, importance=importance, base=base, tag=tag)

    def teach(self, *, example=None, label=None, importance=None, base=None, tag=None, 
              features=None, namespaces=None):
        line = self.make_line(example=example, label=label, importance=importance,
                              tag=tag, features=features, namespaces=namespaces)
        self.send_line(line)

    def _get_prediction_for_line(self, line):
        return self._prediction_factory(self.vw.get_prediction(line))

    def predict(self, *, example=None, base=None, tag=None,
                features=None, namespaces=None):

        line = self.make_line(example=example, no_label=True,
                              tag=tag, features=features, namespaces=namespaces)

        return self._get_prediction_for_line(line)


class OfflineRabbit(Rabbit):
    """
    An OfflineRabbit instance writes whatever examples it is given, in cooked 
    form, in the given file.
    """
    def __init__(self, fp, **kwargs):
        """
        :param fp: an open file, where the examples are written as plain string lines
        
        Other parameters as per Rabbit
        """

        super(Rabbit, self).__init__(kwargs)
        self.fp = fp

    def start(self):
        """
        No-op for OfflineRabbit
        """
        pass

    def predict(self):
        raise Exception("Unable to predict without an actual VW instance")

    def send_line(self, line):
        self.fp.write(line + '\n')

class ActiveRabbit(object):
    def __init__(self, **kwargs):
        super(ActiveRabbit, self).__init__(**kwargs)
        self.options['active'] = True

