import click
import csv

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
def process_file(input_file, output_file):
    """
    A script that takes an input file and converts it from:
    1	!	1814
    to:
    !,1814
    Words with commas inside are preserved.
    """
    # Open the input file for reading
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter='\t', quoting=csv.QUOTE_NONE)

        # Open the output file for writing
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_MINIMAL)

            # Process each row in the input file
            for row in reader:
                if len(row) == 3:
                    # Join the second and third columns with a comma
                    writer.writerow([row[1], row[2]])


if __name__ == '__main__':
    process_file()
