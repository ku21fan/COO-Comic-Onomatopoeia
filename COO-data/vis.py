import os
import math
import argparse
import shutil
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, render_template
import manga109api

from shapely.geometry import Polygon

parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="0.0.0.0", help="host address for webserver"
)
parser.add_argument("--port", type=int, default=6006, help="port for webserver")
parser.add_argument("--debug", action="store_true", help="for debug mode")
parser.add_argument(
    "--static_folder", type=str, default="./vis/", help="static_folder path for flask"
)
parser.add_argument(
    "--template_folder",
    type=str,
    default="./vis/",
    help="template_folder path for flask",
)
parser.add_argument(
    "--manga109_root_dir", type=str, default=".", help="root dir for manga109api",
)
parser.add_argument(
    "--max_page_number",
    type=int,
    default=184,
    help='the maximum page number of manga109 datasets is 184 from the manga "hamlet".',
)
parser.add_argument(
    "-t", "--thickness", type=int, default=5, help="thickness for rectangle/polygon"
)
args = parser.parse_args()

app = Flask(
    __name__, static_folder=args.static_folder, template_folder=args.template_folder
)

# Cache seems to prevent image update on the webserver.
# So to workaround cache problem, use a temporary folder and make a new image every time.
annotated_image_folder = os.path.join(args.static_folder, "tmp")
if os.path.exists(annotated_image_folder):
    shutil.rmtree(annotated_image_folder)
os.makedirs(f"{annotated_image_folder}")

# use manga109api parser
manga109_parser = manga109api.Parser(root_dir=args.manga109_root_dir)

with open(f"{args.manga109_root_dir}/books.txt", "r") as book_list:
    manga_list = book_list.readlines()
page_list = list(range(0, args.max_page_number + 1))


@app.route("/", methods=["GET", "POST"])
def visualization():
    if request.method == "POST":
        manga_name = request.form["manga_name"]

        # for multiple visualizations -> typing '2-5' shows images from 002.jpg to 005.jpg.
        # e.g. "2" -> [2].   "2-5" -> [2, 3, 4, 5]
        page_index_list = request.form["page_index"].split("-")
        page_index_list = list(
            range(int(page_index_list[0]), int(page_index_list[-1]) + 1)
        )

        parser_anno_type = "annotations"
        annotation = manga109_parser.get_annotation(
            book=manga_name, annotation_type=parser_anno_type
        )

        font = ImageFont.truetype(
            f"{args.static_folder}/NotoSansJP-Bold.otf", int(request.form["font_size"])
        )
        top_margin = int(request.form["top_margin"])
        right_margin = int(request.form["right_margin"])

        annotated_image_path_list = []
        for page_index in page_index_list:
            image_path = manga109_parser.img_path(book=manga_name, index=page_index)

            try:
                image = Image.open(image_path)
            except:
                print(f"There is no image file at {image_path}")
                continue

            width, height = image.size
            image_margin = Image.new(
                "RGB", (width + right_margin, height + top_margin), color="white"
            )
            image_margin.paste(image, (0, top_margin, width, height + top_margin))
            draw = ImageDraw.Draw(image_margin)

            if "onomatopoeia" in request.form or "onomatopoeia_wo_text" in request.form:
                annotation_type_list = [
                    "onomatopoeia",
                    "onomatopoeia_link1",
                    "onomatopoeia_link2",
                ]
                color_dict = {
                    "onomatopoeia": "green",
                    "onomatopoeia_link1": "#8a2be2",
                    "onomatopoeia_link2": "orange",
                }
                onomatopoeia_center_dict = {}

                # rendering annotations on the original image
                for annotation_type in annotation_type_list:
                    try:
                        rois = annotation["page"][page_index][annotation_type]
                        if isinstance(rois, dict):
                            rois = [rois]  # for one instance case.
                    except:
                        continue

                    for roi in rois:
                        if annotation_type == "onomatopoeia":
                            x_list = [int(roi[attr]) for attr in roi if "@x" in attr]
                            y_list = [
                                int(roi[attr]) + top_margin
                                for attr in roi
                                if "@y" in attr
                            ]

                            # keep onomatopoeia center for onomatopoeia_link
                            center_x = int(sum(x_list) / len(x_list))
                            center_y = int(sum(y_list) / len(y_list))
                            onomatopoeia_center_dict[roi["@id"]] = (center_x, center_y)

                            x_list.append(x_list[0])  # to use draw.line() for polygon
                            y_list.append(y_list[0])  # to use draw.line() for polygon
                            polygon = [(x, y) for x, y in zip(x_list, y_list)]

                            # draw.polygon() can not adjust thickness of line, so we use draw.line()
                            draw.line(
                                polygon,
                                fill=color_dict[annotation_type],
                                width=args.thickness,
                            )

                            # Polygon_object = Polygon(polygon[:-1])
                            # draw.line(Polygon_object.minimum_rotated_rectangle.exterior.coords,
                            #           fill='orange', width=args.thickness)

                            # write text at annotation staring point.
                            text = roi["#text"]
                            text_width, text_height = font.getsize(text)
                            x = x_list[0]
                            y = y_list[0]
                            x_last = x_list[-2]
                            y_last = y_list[-2]
                            point_size = 6
                            draw.rectangle(
                                (
                                    x - point_size,
                                    y - point_size,
                                    x + point_size,
                                    y + point_size,
                                ),
                                fill="red",
                            )  # start point
                            draw.rectangle(
                                (
                                    x_last - point_size,
                                    y_last - point_size,
                                    x_last + point_size,
                                    y_last + point_size,
                                ),
                                fill="blue",
                            )  # end point

                            if "onomatopoeia_wo_text" in request.form:
                                pass
                            else:
                                draw.rectangle(
                                    (x, y - text_height, x + text_width, y),
                                    fill="white",
                                )
                                draw.text(
                                    (x, y - text_height),
                                    text,
                                    font=font,
                                    fill=color_dict[annotation_type],
                                )

                        elif "onomatopoeia_link" in annotation_type:
                            link_id_list = [
                                roi[attr] for attr in roi if "@link" in attr
                            ]
                            # print(link_id_list)

                            # TODO: draw link arrow? now only line
                            for i, link_id in enumerate(link_id_list):

                                if i == 0:
                                    point_size = 10
                                    center_x = onomatopoeia_center_dict[link_id][0]
                                    center_y = onomatopoeia_center_dict[link_id][1]
                                    draw.rectangle(
                                        (
                                            center_x - point_size,
                                            center_y - point_size,
                                            center_x + point_size,
                                            center_y + point_size,
                                        ),
                                        fill="red",
                                    )  # start point
                                    link_line = [(center_x, center_y)]

                                elif i == len(link_id_list) - 1:
                                    point_size = 10
                                    center_x = onomatopoeia_center_dict[link_id][0]
                                    center_y = onomatopoeia_center_dict[link_id][1]
                                    draw.rectangle(
                                        (
                                            center_x - point_size,
                                            center_y - point_size,
                                            center_x + point_size,
                                            center_y + point_size,
                                        ),
                                        fill="blue",
                                    )  # end point

                                    link_line.append(onomatopoeia_center_dict[link_id])
                                else:
                                    link_line.append(onomatopoeia_center_dict[link_id])

                            draw.line(
                                link_line, fill=color_dict[annotation_type], width=8
                            )

                        else:
                            # draw region of interest.
                            draw.rectangle(
                                [
                                    roi["@xmin"],
                                    roi["@ymin"] + top_margin,
                                    roi["@xmax"],
                                    roi["@ymax"] + top_margin,
                                ],
                                outline=color_dict[annotation_type],
                                width=args.thickness,
                            )

                            # show character name
                            if annotation_type in {"body", "face"}:
                                text = char_id_to_name[roi["@character"]]
                                text_width, text_height = font.getsize(text)
                                x = roi["@xmin"]
                                y = roi["@ymin"] + top_margin
                                draw.rectangle(
                                    (x, y - text_height, x + text_width, y),
                                    fill="white",
                                )
                                draw.text(
                                    (x, y - text_height),
                                    text,
                                    font=font,
                                    fill=color_dict[annotation_type],
                                )

                # To workaround cache problem, make a new image every time by using _{datetime.now().microsecond}.
                annotated_image_path = f"{annotated_image_folder}/annotated_{manga_name}_page{page_index}_{datetime.now().microsecond}.jpg"
                image_margin.save(annotated_image_path)
                annotated_image_path_list.append(annotated_image_path)

            elif "image_only" in request.form:
                # To workaround cache problem, make a new image every time by using _{datetime.now().microsecond}.
                annotated_image_path = f"{annotated_image_folder}/annotated_{manga_name}_page{page_index}_{datetime.now().microsecond}.jpg"
                image_margin.save(annotated_image_path)
                annotated_image_path_list.append(annotated_image_path)

        return render_template(
            "vis.html",
            annotated_image_list=annotated_image_path_list,
            manga_list=manga_list,
            page_list=page_list,
            manga_name=request.form["manga_name"],
            page_index=request.form["page_index"],
            image_height=request.form["image_height"],
            font_size=request.form["font_size"],
            top_margin=request.form["top_margin"],
            right_margin=request.form["right_margin"],
        )

    else:
        return render_template(
            "vis.html",
            manga_list=manga_list,
            page_list=page_list,
            image_height=1000,
            font_size=20,
            top_margin=0,
            right_margin=0,
        )


if __name__ == "__main__":
    app.run(host=args.host, port=args.port, debug=args.debug)
