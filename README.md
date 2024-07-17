# Duplicate Contrastive Explanation

Reference from paper [ContrXT: Generating contrastive explanations from any text classifier](https://www.sciencedirect.com/science/article/abs/pii/S1566253521002426)

## Install
1. Connect conda
```sh
D:

cd D:\Desktop\xAI\exp\22C15033_ContrXT
conda env list

conda activate 'env_conda'
```

### Run demo

```sh
sh scripts/demo/demo.sh

python scripts/demo/demo.py
```


### Run on server
1. Connect virtual enviroment
```sh
conda activate khoaha_venv
```

2. List conda enviroment
```sh
conda env list
```

3. Install `pip` in Conda
```sh
conda install pip
```


4. Install Graphviz
```sh
sudo apt-get install graphviz
```

5. Experiment with test classification
```sh
sh scripts/text_cls/run_text_cls.sh
```


```sh
conda activate khoaha_venv
```


## Venv local

1. Activate
```sh
source .venv/Scripts/activate
```

```sh
python src/text_cls/run.py -n -l vi -p src/text_cls/dataset/VNTC/original/train.csv

python src/text_cls/run.py -n -l en -p src/text_cls/dataset/20newsgroups/orginal/train_part_1.csv

python src/text_cls/run.py -n -l en -p src/text_cls/dataset/20newsgroups/orginal/test_part_1.csv -t

python src/text_cls/run.py -n -r -l vi -p src/text_cls/dataset/VNTC/original/train.csv
```


## Regrex

```sh
[\+]?[0-9]?[-\s\.]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6} -> space
Thanx
\s\s+
~
--+
(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])
\n\n+
\n 
\t\t+
\^\^+
==+
\*\*+
(From article by)[\s\(\) \tA-Za-z0-9:!"#$%&\*\+,./:;<=>?@^_`{|}~]*
(From:|Nntp-Posting-Host:|X-X-From:|Date:|Cc:|Reply-To:|In-Reply-To:|To:|)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~]*
(From:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Nntp-Posting-Host:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Date:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Cc:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Reply-To:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(In-Reply-To:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Newsgroups:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Status:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Re:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(News-Software:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Organization:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Lines:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Message-Id:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Sender:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Distribution:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Keywords:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Expires:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Received:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(X-Charset:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Postal Zone:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Content-Transfer-Encoding)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Content-Length:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Content-Type:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-]*
(Mime-Version:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-\[\]\{\}]*
(Internet:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-\[\]\{\}]*
(AOL:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-\[\]\{\}]*
(Fedex:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-\[\]\{\}]*
(FAX:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-\[\]\{\}]*
(PHONE:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-\[\]\{\}]*
(Voice)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-\[\]\{\}]*
(CompuServe:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-\[\]\{\}]*
(Phishnet:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-\[\]\{\}]*
(Summary:)[\s\(\) \tA-Za-z0-9:!#$%&\*\+,./:;<=>?@^_`{|}~\-\[\]\{\}]*
^--\n
^-- \n
^--*\n
^---\n
^----\n
^-----\n
^------\n
^----------------------------------------------------------------------\n
Archive-name:
/ hpcc01:rec.motorcycles /  (John Stafford) / 11:06 am  Apr  1, 1993 /
Let me see, ""unless you have an accident, you won't need more"", hmmmmmmm.
^==*\n
^  --*\n
-Eric
DoD
- David
-Tim May
[deleted]
Dave.
Thanks\s*\n
Adam Shostack
-Bryan
From article  by  
reply to
[stuff deleted]
Kristen
Karl R. Schimmel
End of part
thanks\n
Thanks in advance.
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
Nanci
Kent
--Cliff
jamie
Serdar Argic
Hasta luego
- Erwin
-John Neuharth
Tom Nguen
Mick Mueck 
luigi
Burkhard Neidecker-Lutz
Joerg Haber
Stephen

Stephen

    _/_/_/_/  _/_/_/_/   _/       _/    * Atheist
   _/        _/    _/   _/ _/ _/ _/     * Libertarian
  _/_/_/_/  _/_/_/_/   _/   _/  _/      * Pro-individuality
       _/  _/     _/  _/       _/       * Pro-responsibility
_/_/_/_/  _/      _/ _/       _/ Jr.    * and all that jazz...

Mike

Vesselin

Gerard.

Link Hudson

Jouni

Rob Lansdale

Jae

Cathy

Andy Byler

jason

^\d+,

Jack Waters II
DoD#1919

David B. Lewis   
 ""Just the FAQs, ma'am."" -- Joe Friday 
Duke.
- Thomas
jerry
Dani Eder
 brian
 Sean
 John
tont
marc
euclid
Kent
Stephen
Gary
James Kiefer
Ken
Mike
Doug
Naomi
KOZ
Greg.
Stefan
dave
-Joe Betts
George
Nigel
Todd
Grant
jsh
Dave
--Craig
 Mike Smith
Van Kelly
Charles Campbell
Rob
-Billy
Naftaly
--rich
Stuart
Greg Greene
Larry Overacker
Nancy M


Tới 617
stuff deleted
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
Shalom,                                  Len Howard
Bill Guilford
Sam
David Bull
insert deletion
Mike Ruff\n

Followup-To
Supersedes:
Archive-name:
Last-modified:
AOL:
Fedex:
PHONE:
FAX
Voice
CompuServe:
Factsnet:
Phishnet:
Summary:
Expires:
Received:
Path:
Followup-To:

Greetings,
Cheers
Thank

[ Article crossposted from comp.windows.ms ]
[ Author was Kevin Routh ]
[ Posted on 19 Apr 1993 12:35:55 GMT ]


godiva.nectar.cs.cmu.edu:/usr0/anon/pub/firearms/politics/rkba/Congress

From x51948b1@usma1.USMA.EDU Tue Apr 20 10:28:47 1993
Received: from usma1.usma.edu by trotter.usma.edu (4.1/SMI-4.1-eef)
	id AA01628; Tue, 20 Apr 93 11:27:50 EDT
Received:  by usma1.usma.edu (5.51/25-eef)
	id AA03219; Tue, 20 Apr 93 11:20:18 EDT
Message-Id: < .AA03219@usma1.usma.edu>
Date: Tue, 20 Apr 93 11:20:17 EDT
From: x51948b1@usma1.USMA.EDU (Peckham David CDT)
To: cs1442au@decster.uta.edu
Subject: Problem.
Status: OR

```