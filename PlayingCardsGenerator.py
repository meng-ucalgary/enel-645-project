from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import numpy as np

class CardsDirectoryIterator(DirectoryIterator):
    
    suits_names = np.array(["clubs","diamonds","hearts","spades"])
    values_names = np.array(["2","3","4","5","6","7","8","9","10","ace","jack","king","queen"])
        
    list_of_labels =np.array([
        ('2', 'clubs'), ('2', 'diamonds'), ('2', 'hearts'), ('2', 'spades'),
        ('3', 'clubs'), ('3', 'diamonds'), ('3', 'hearts'), ('3', 'spades'),
        ('4', 'clubs'), ('4', 'diamonds'), ('4', 'hearts'), ('4', 'spades'),
        ('5', 'clubs'), ('5', 'diamonds'), ('5', 'hearts'), ('5', 'spades'),
        ('6', 'clubs'), ('6', 'diamonds'), ('6', 'hearts'), ('6', 'spades'),
        ('7', 'clubs'), ('7', 'diamonds'), ('7', 'hearts'), ('7', 'spades'),
        ('8', 'clubs'), ('8', 'diamonds'), ('8', 'hearts'), ('8', 'spades'),
        ('9', 'clubs'), ('9', 'diamonds'), ('9', 'hearts'), ('9', 'spades'),
        ('10', 'clubs'), ('10', 'diamonds'), ('10', 'hearts'), ('10', 'spades'),
        ('ace', 'clubs'), ('ace', 'diamonds'), ('ace', 'hearts'), ('ace', 'spades'),
        ('jack', 'clubs'), ('jack', 'diamonds'), ('jack', 'hearts'), ('jack', 'spades'),
        ('king', 'clubs'), ('king', 'diamonds'), ('king', 'hearts'), ('king', 'spades'),
        ('queen', 'clubs'), ('queen', 'diamonds'), ('queen', 'hearts'), ('queen', 'spades')])
    
    def __init__(self, directory, image_data_generator,
               target_size=(256, 256),
               color_mode='rgb',
               classes=None,
               class_mode='categorical',
               batch_size=32,
               shuffle=True,
               seed=None,
               data_format=None,
               save_to_dir=None,
               save_prefix='',
               save_format='png',
               follow_links=False,
               subset=None,
               interpolation='nearest',
               dtype=None):
        
        #class the superclass __init__
        super().__init__(
               directory,
               image_data_generator,
               target_size,
               color_mode,
               classes,
               class_mode,
               batch_size,
               shuffle,
               seed,
               data_format,
               save_to_dir,
               save_prefix,
               save_format,
               follow_links,
               subset,
               interpolation,
               dtype)
    
    def get_labels(self, idx):
        return self.list_of_labels[idx]
    
    def create_value_one_hot(self, value):

        one_hot_value = np.zeros(len(self.values_names))

        #sets the idx that corresponds to the value to 1
        one_hot_value[np.where(self.values_names == value)] = 1
        
        return one_hot_value
    
    def create_suit_one_hot(self, suit):

        one_hot_suit = np.zeros(len(self.suits_names))

        #sets the idx that corresponds to the suit to 1
        one_hot_suit[np.where(self.suits_names == suit)] = 1
        
        return one_hot_suit
    
    # override the __getitem__ method so it returns two arrays of labels
    # one for the suit and the other one for the value
    def __getitem__(self, idx):
        
        samples, labels = super().__getitem__(idx)
        
        batch_size = samples.shape[0]

        suits_array = np.expand_dims(np.zeros(len(self.suits_names) * batch_size), axis=0)
        values_array = np.expand_dims(np.zeros(len(self.values_names) * batch_size), axis=0)
    
        samples_suits_one_hot = np.reshape(suits_array, (batch_size, len(self.suits_names)))
        samples_values_one_hot = np.reshape(values_array, (batch_size, len(self.values_names)))
    
        # for each sample in the batch, create the two one-hot encoded arrays with the suit label
        # and value label
        for i in range(batch_size):

            unified_one_hot_label = labels[i,:]
            
            value, suit = self.get_labels(np.argmax(unified_one_hot_label))
   
            one_hot_value = self.create_value_one_hot(value)
            one_hot_suit = self.create_suit_one_hot(suit)
            
            samples_suits_one_hot[i] = one_hot_suit
            samples_values_one_hot[i] = one_hot_value

        return samples, (samples_suits_one_hot, samples_values_one_hot)

class CardsDataGenerator(ImageDataGenerator):
    
    def flow_from_directory(self,
                          directory,
                          target_size=(256, 256),
                          color_mode='rgb',
                          classes=None,
                          class_mode='categorical',
                          batch_size=32,
                          shuffle=True,
                          seed=None,
                          save_to_dir=None,
                          save_prefix='',
                          save_format='png',
                          follow_links=False,
                          subset=None,
                          interpolation='nearest'):
            
        dir_iterator_class = super().flow_from_directory(
                          directory,
                          target_size,
                          color_mode,
                          classes,
                          class_mode,
                          batch_size,
                          shuffle,
                          seed,
                          save_to_dir,
                          save_prefix,
                          save_format,
                          follow_links,
                          subset,
                          interpolation)
        
        # casts the DirectoryIterator class returned by the flow_from_directory method
        # to the CardsDirectoryIterator class defined above
        dir_iterator_class.__class__ = CardsDirectoryIterator
        
        return dir_iterator_class