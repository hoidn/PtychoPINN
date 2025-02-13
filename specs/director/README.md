## example usage
run this to generate json formatted list of files
python dependency_detector.py --context-file context --task tprompt.txt 

then, something like:
 python director.py --config specs/director_bugfix_generic.yaml --dependencies dependencies.json --template-values tprompt2.yaml


