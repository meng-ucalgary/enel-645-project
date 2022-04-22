import cv2
import argparse
import tensorflow as tf
from poker import Range
from poker.hand import Combo
import sys
sys.path.append("./holdem_calc")
import holdem_functions
import holdem_calc

classes_names = ["2c", "2d", "2h", "2s",
                 "3c", "3d", "3h", "3s",
                 "4c", "4d", "4h", "4s",
                 "5c", "5d", "5h", "5s",
                 "6c", "6d", "6h", "6s",
                 "7c", "7d", "7h", "7s",
                 "8c", "8d", "8h", "8s",
                 "9c", "9d", "9h", "9s",
                 "Tc", "Td", "Th", "Ts",
                 "Ac", "Ad", "Ah", "As",
                 "Jc", "Jd", "Jh", "Js",
                 "Kc", "Kd", "Kh", "Ks",
                 "Qc", "Qd", "Qh", "Qs"]


def get_cards_from_image(filename):

    image = cv2.imread(filename)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # trial and error to obtain the best values here
    edges = cv2.Canny(image, 180, 280)

    # find all the external contours in the image (external contours is defined by the cv2.RETR_EXTERNAL flag)
    contours, _ = cv2.findContours(
        edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort the contours using the enclosed area
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # get only the first 9 greatest contours, in our case, we are dealing with only 2 players
    # so at maximum we will have 9 cards to extract, 2 from each player plus up to 5 from the table
    biggest_countours = sorted_contours[:9]

    # get the rectangles from the contours
    rectangles = []
    min_width_px = 20
    min_height_px = 20
    for contour in biggest_countours:
        rect = cv2.boundingRect(contour)

        # only add rectangles above the minimum size threshold
        if rect[2] > min_width_px and rect[3] > min_height_px:
            rectangles.append(rect)

    # removes duplicates that are sometimes returned
    rectangles = list(set(rectangles))

    img_contour = image.copy()
    extracted_cards = []
    for rect in rectangles:

        # gets the rectangle edgs coordinates
        x_top_left, y_top_left, width, height = rect[0], rect[1], rect[2], rect[3]

        # crops the cards from the image
        cropped_image = img_contour[y_top_left: (
            y_top_left + height), x_top_left: (x_top_left + width), :]

        # if the width is greater than the height, rotate 90 degs, so all the cards are up
        if cropped_image.shape[0] < cropped_image.shape[1]:
            cropped_image = cv2.rotate(
                cropped_image, cv2.cv2.ROTATE_90_CLOCKWISE)

        coords = ({"x_top_left": x_top_left, "y_top_left": y_top_left,
                  "width": width, "height": height})

        extracted_cards.append((cropped_image, coords))

    return extracted_cards


def make_combo(cards):

    combo = ""
    for card in cards:
        combo += card

    return combo


def classify_cards(extracted_cards, model):

    hero_cards = []
    villan_cards = []
    table_cards = []

    for card in extracted_cards:

        card_image = card[0]
        card_coordinate = card[1]

        resized = cv2.resize(card_image, dim)
        resized = resized[tf.newaxis, ...]

        Ypred = model.predict(resized)
        index = Ypred.argmax()

        # player 1 cards are to the left of the image
        if card_coordinate['x_top_left'] < hero_threshold:
            hero_cards.append(classes_names[index])

        # player 2 cards are to the left of the image
        elif card_coordinate['x_top_left'] > villan_threshold:
            villan_cards.append(classes_names[index])

        # all the other cards, are table cards
        else:
            table_cards.append(classes_names[index])

    return hero_cards, villan_cards, table_cards


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Poker Odds calculator')
    parser.add_argument(
        'image', help='original image to extract the cards from')
    parser.add_argument('model', help='model to be used')

    args = parser.parse_args()

    extracted_cards = get_cards_from_image(args.image)

    model = tf.keras.models.load_model(args.model)

    # retrieves the model input size
    config = model.get_config()
    width = config['layers'][0]['config']['batch_input_shape'][1]
    height = config['layers'][0]['config']['batch_input_shape'][2]

    dim = (width, height)

    original_image_width = cv2.imread(args.image).shape[1]

    # thresholds to define to which player each card belongs
    # The hero cards will be on the leftmost part of the image,
    # the villan cards will be on the rightmost part of the image.
    # Table cards will be in the middle portion of the image
    hero_threshold = original_image_width * 0.3
    villan_threshold = original_image_width * 0.6

    hero_cards, villan_cards, table_cards = classify_cards(
        extracted_cards, model)

    # first 3 cards are flop
    flop = table_cards[:3]

    board = table_cards

    print("Hero hand = {}".format(hero_cards))
    print("Villan hand = {}".format(villan_cards))
    print("Flop = {}".format(flop))
    print("Board = {}".format(board))

    hero_hand = Combo(make_combo(hero_cards))
    villan_hand = Combo(make_combo(villan_cards))

    exact_calculation = True
    verbose = True

    odds = holdem_calc.calculate_odds_villan(board, exact_calculation,
                                             1, None,
                                             hero_hand, villan_hand,
                                             verbose, print_elapsed_time=False)

    print("Odds of hero winning: {:1.2f}%".format(odds[0]['win'] * 100))
    print("Odds of hero losing: {:1.2f}%".format(odds[0]['lose'] * 100))
    print("Odds of tie: {:1.2f}%".format(odds[0]['tie'] * 100))
