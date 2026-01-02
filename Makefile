FLAGS = --pdf-engine=xelatex -V geometry:margin=1in --filter mermaid-filter

all: parallelism.pdf

parallelism.pdf: README.md
	pandoc $< -o $@ $(FLAGS)

clean:
	rm mermaid-filter.err

.PHONY: all clean

