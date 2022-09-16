import difflib
import sys

if __name__ == "__main__":

    with open("books.txt", "r") as bookfile:
        booklist = bookfile.read().splitlines()

    for book in booklist:
        fromfile = f"annotations/{book}.xml"
        tofile = f"public-annotations/COO-Comic-Onomatopoeia/v2022.07.07/{book}.xml"

        with open(fromfile) as ff:
            fromlines = ff.readlines()
        with open(tofile) as tf:
            tolines = tf.readlines()

        print(
            f"Compare annotations of {book:25s}\t# of lines from COO repo annotation: {len(fromlines):4d}\tfrom public-annotations repo {len(tolines):4d}"
        )

        diff = difflib.unified_diff(
            fromlines, tolines, fromfile=fromfile, tofile=tofile
        )
        sys.stdout.writelines(diff)
