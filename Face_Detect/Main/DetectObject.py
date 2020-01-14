import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import math

class DetectObject():
    def __init__(self):
        self.face_encoding_list = []
        self.face_names = []
    def pre_Train(self, fileName):
        with open(fileName, 'rb') as f:
            self.face_encoding_list, self.face_names = pickle.load(f)

    def face_distance_to_conf(self, face_distance, face_match_threshold=0.6):
        if face_distance > face_match_threshold:
            range = (1.0 - face_match_threshold)
            linear_val = (1.0 - face_distance) / (range * 2.0)
            return linear_val
        else:
            range = face_match_threshold
            linear_val = 1.0 - (face_distance / (range * 2.0))
            return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))


    def Detect_Object(self, fileName, deepCheck):
        ret_msg = ""
        name = ""
        acc = 0
        # Load an image with an unknown face
        unknown_image = face_recognition.load_image_file(fileName)

        # Find all the faces and face encodings in the unknown image
        if deepCheck == 2:
            face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=0, model="cnn")
        else:
            face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

        # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
        # See http://pillow.readthedocs.io/ for more about PIL/Pillow
        pil_image = Image.fromarray(unknown_image)
        # Create a Pillow ImageDraw Draw instance to draw with
        draw = ImageDraw.Draw(pil_image)

        # Loop through each face found in the unknown image
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.face_encoding_list, face_encoding)

            name = "Unknown"
            acc = 0

            # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.face_encoding_list, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.face_names[best_match_index]
                acc = self.face_distance_to_conf(face_distances[best_match_index])
            ret_msg += "Name: " + name + "\nAccuracy: " + str(acc * 100) + "%\n"
            name += "-" + str(acc * 100) + "%"

            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

            # Draw a label with a name below the face
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        # Remove the drawing library from memory as per the Pillow docs
        del draw

        # Display the resulting image
        #pil_image.show()

        # You can also save a copy of the new image to disk if you want by uncommenting this line
        pil_image.save("result.jpg")

        return ret_msg