#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, division

import sys,traceback
import argparse
import bz2
import codecs
import cgi
import fileinput
import logging
import os.path
import re  # TODO use regex when it will be standard
import time
from io import StringIO
from multiprocessing import Queue, Process, Value, cpu_count
from timeit import default_timer
from pattern.en import parse, pprint, WORD, POS, CHUNK, PNP, LEMMA, pprint, parsetree, tokenize

__version__ = '20170614'

PY2 = sys.version_info[0] == 2
if PY2:
    from urllib import quote
    from htmlentitydefs import name2codepoint
    from itertools import izip as zip, izip_longest as zip_longest
    range = xrange  # Overwrite by Python 3 name
    chr = unichr    # Overwrite by Python 3 name
    text_type = unicode
else:
    from urllib.parse import quote
    from html.entities import name2codepoint
    from itertools import zip_longest
    text_type = str


# Program version
version = '20170606'

    

ReTag_sectiontitle = re.compile(r'<h(\d+)>([^<]*?)</h\1>') #<h3>Spanish Revolution</h3>
ReTag_doctitle = re.compile(r'(?:<doc id="[^>]*?">)|(?:</doc>)')#<doc id="12" url="https://en.wikipedia.org/wiki?curid=12" title="Anarchism">
ReTag_categoryline = re.compile(r'\[\[(Category:[^\]]*?)\]\]')#[[Category:Anarchism|]]

ReTag_wikilinkpostfix = re.compile(r'\[\[([^\]\[]*?)\]\]([\'_—–\-\w]+)') #[[Spanish|Communist Party]]-led
ReTag_wikilinkprefix = re.compile(r'([\'_—–\-\w]+)\[\[([^\]\[]*?)\]\]') #non-–[[Hierarchy|hierarchical]]

ReTag_wikilinkpunctuationfix1 = re.compile(r'\[\[([^|\]\[]*?)[\?]\]\]')#[[What is Property?]]
ReTag_wikilinkpunctuationfix2 = re.compile(r'\[\[(([^\]\[]*?)[\?]?\|([^\]\[]*?)[\?]?)\]\]')#[[this is what?|What is Property?]]

ReTag_wikilinkhyphen = re.compile(r'\]\][_—–\-]\[\[') # ]]–[[
ReTag_wikilinksegsentence = re.compile(r'(\[\[[^\]]*\]\][.!,;\?])(\w)')#for [[desegregation]].The civil
reTag_wikilinkpresymbol = re.compile(r'([^\s]?:)(\[\[)') #e Electron include:[[Crystal Castles 

ReTag_wikilink1 = re.compile(r'\[\[([^|\]\[]*?)\]\]')#[[electrical energy]]
ReTag_wikilink2 = re.compile(r'\[\[(([^\]\[]*?)\|([^\]\[]*?))\]\]([\'_—–\-\w]*)') #[[photovoltaic system|photovoltaic device]]s.

ReTag_wikilink1_aftertoken = re.compile(r'\[\[ ([^|\]\[]*?) \]\]')#[[ Presbyterian church ]]
ReTag_wikilink2_aftertoken = re.compile(r'\[\[ (([^\]\[]*?) \| ([^\]\[]*?)) \]\]')#[[ Presbyterian Church in America | PCA ]]
 
ReTag_wikilink1_r = re.compile(r'\[ \[ ([^|\]\[]*?) \] \]')#[ [ electrical energy ] ]
ReTag_wikilink2_r = re.compile(r'\[ \[ (([^\]\[]*?) \| ([^\]]*?)) \] \]')#[ [ photovoltaic system | photovoltaic device ] ]

ReTag_bracketleft = re.compile(r'\[\[')
ReTag_bracketright = re.compile(r'\]\]')

ReTag_space = re.compile(r' {2,}')
ReTag_blankwikilink = re.compile(r'\[\[ *\]\]')

ReTag_x85etc = re.compile(r'\x85|\u015b|\u2029|\u03b8') # replace especial character
ReTag_tokenizebug1 = re.compile(r':-\[ \[')# fix the bug of tokenize, eg: 'Executed :- [[ Carl Hans Lody ]]' -->  'Executed :-[ [ Carl Hans Lody ] ]'
ReTag_tokenizebug2 = re.compile(r':\[ \[')

ReTag_nowiki= re.compile(r'<nowiki>(.*?)</nowiki>')#<nowiki>to [buy [a car]]</nowiki>

ReTag_checkret_2rightbracket = re.compile(r'^([^\] ]*?)\]\][^/]*?/([^/]*?)/([^\] ]*?)\]\][^/]*?$')#le/VBP/le Carré]]-style/JJ/carré]]-style spy/NN/spy
ReTag_checkret_vertlineleftbracket = re.compile(r'^([^\] ]*?)\|\[[^/]*?/([^/]*?)/([^\] ]*?)\|\[[^/]*?$')#le/NN/le Carré|[John/NNP/carré|[john ]/)/] le/VBP/le
ReTag_checkret_3blackslash = re.compile(r'^//[^/]*?/$')# //CC/
ReTag_checkret_2leftbracket = re.compile(r'^\[\[/([^/]*?)/\[\[$')#[[/VB/[[
ReTag_checkret_2bracket = re.compile(r'(\]\])|(\[\[)')
#ReTag_checkret_beginendwithslash = re.compile(r'^/[^ ]*?/$') #remove '/><section/NN/' [/(/[ /NN/ ]/)/] |/NN/|
ReTag_checkret_beginendwithslash = re.compile(r'^(/|\[|\]|\|)[^ ]*?(\1)$') #remove '/><section/NN/' [/(/[ /NN/ ]/)/] |/NN/|
ReTag_checkret_morethan2slash = re.compile(r'\/')#Alumni</sub></h2>/NNP/alumni<

ReTag_checkret_21 = re.compile(r'( ?\[\[ \| [^\]]*? \]\] ?)')      # [[ | x/x/x ]]  替换为1个空格)
ReTag_checkret_22 = re.compile(r'( ?\[\[ [^\[]*? \| \]\] ?)')      #  [[ x/x/x | ]]  替换为1个空格
ReTag_checkret_23 = re.compile(r'( ?\[\[ \| \]\] ?)')     #  [[ | ]]  替换为1个空格
ReTag_checkret_24 = re.compile(r'( ?\[\[ +\]\] ?)')
ReTag_checkret_25 = re.compile(r'( +)') #将多个连续空格替换为1个

def Parse(input_file, output_file):
    '''
    Parese the data in inputfile, and save the result into outputfile
    :param inputfile: the inputfile, which has been opened by FileInput. 
                      The formattion of the file should be as Lustyle
    :param outputfile: the outputfile, which has been opened or std.out
    '''
    inputfile = fileinput.FileInput(input_file, openhook=fileinput.hook_compressed)
        
    if output_file == 'text': #this means default value of '--output'
        outputfile = sys.stdout
    else:
        outputfile = open(output_file,'w')    
    
    lineno = 1
    
    for line in inputfile:      
        if lineno == 1:
            if outputfile == sys.stdout:
                print '=========================================',
            print 'filename:',
            print inputfile.filename(),
            print ' begin to be processed ... '
        #if lineno < 3281:
        #    lineno += 1
        #    continue
        #print line
        line = line.decode('utf-8')# this is important for avoding the error of Unicode decode
        #line = ' - Executed :- [[Carl Hans Lody]] on 6 November 1914, in the Miniature Rifle Range.'
        if outputfile == sys.stdout:
            print '==================='        
            print 'input file line no:',
            print lineno
            print '-----------\nline:',
            #print line
            print line.encode('utf-8') #avoid the error on the cluster server
        else:
            if lineno % 100000 == 0 or lineno == 1:
                print 'parse_process pid:{}, {}: {}'.format(os.getpid(),os.path.basename(input_file),lineno)
        lineresult = ParseOneLine(line,inputfile)
        if outputfile == sys.stdout:
            lineresult = lineresult.encode('utf-8')  #avoid the error on the cluster server
            print '-----------\nlineresult:',
            print lineresult
        else:
            lineresult = (lineresult+'\n').encode('utf-8')
            outputfile.write(lineresult)
            
        #outputfile.
        lineno += 1
    
    print 'parse_process pid:{}, {}: {}'.format(os.getpid(),os.path.basename(input_file),lineno)    
    inputfile.close()
    if outputfile != sys.stdout:
        outputfile.close()
        os.rename(output_file, output_file+"Parsed")
    
    
    if outputfile == sys.stdout:
        print '=========================================',
    print 'filename:',
    print inputfile.filename(),
    print ' has been processed successfully!'
    return 



def MergeParseResultofOneSentence(token_sen, dispformstree_sen, internalformstree_sen, inputfile):
    '''
    merge the parse results of dispformstree_sen and internalformstree_sen,
    so as to get the parse result of token_sen
    '''
    wordsintoken_sen = token_sen.split(' ')

    taggedsen_byme = []
    pos_dispforms = 0
    pos_internalforms = 0
    pos_wordsintoken_sen = 0
    word = ''

    try:
        while pos_wordsintoken_sen < len(wordsintoken_sen):
        #for i in range(0,len(wordsintoken_sen),1): #range（0,5,1） [0, 1, 2, 3, 4]
            word = wordsintoken_sen[pos_wordsintoken_sen]
            #print word
            
                    
            if word != '[[':#    and word != ']]' and word != '|':
                # ']]' and '|' can be processed in the block of word == '[['
                dispword = dispformstree_sen[pos_dispforms]
                taggedsen_byme.append(dispword.string+'/'+dispword.type+'/'+dispword.lemma)
                #print taggedsen_byme
                
                pos_dispforms += 1  #the dispword emerges in dispforms, internalforms and token_sen, so add all of them
                pos_internalforms += 1
                pos_wordsintoken_sen += 1
                continue
            
            if word == '[[': # handle the whole wiki link mark
                taggedsen_byme.append(u'[[')
                pos_wordsintoken_sen += 1#the word only emerges in token_sen, so add it only
                
                posnextvline = findpos('|', wordsintoken_sen, pos_wordsintoken_sen)
                posnextrightbrackets = findpos(']]', wordsintoken_sen, pos_wordsintoken_sen)            
                
                if posnextrightbrackets == None:
                    pass
                    print 'there should be a balanced right brackets, but Not!'
                    print 'this can\'t be handled by current version'
                    print 9/0
                elif posnextvline == None or posnextvline > posnextrightbrackets:
                    #means [[ a a ]]
                    pass
                    #pos_wordsintoken_sen += 1
                    #append the parse of the part
                    len_part = posnextrightbrackets - (pos_wordsintoken_sen-1) -1
                    l=0
                    while l < len_part:
                        dispword = dispformstree_sen[pos_dispforms]
                        taggedsen_byme.append(dispword.string+'/'+dispword.type+'/'+dispword.lemma)
                        pos_dispforms += 1 #the dispword emerges in dispforms, internalforms and token_sen, so add all of them
                        pos_internalforms += 1
                        pos_wordsintoken_sen += 1
                        l += 1                
                elif posnextvline < posnextrightbrackets:
                    #means [[ a a | b b ]]
                    len_leftpart = posnextvline - (pos_wordsintoken_sen-1) -1
                    len_rightpart = posnextrightbrackets - posnextvline -1
                    #appent the parse of left part
                    l=0
                    while l < len_leftpart:
                        internalword = internalformstree_sen[pos_internalforms]
                        taggedsen_byme.append(internalword.string+'/'+internalword.type+'/'+internalword.lemma)
                        pos_internalforms += 1 #the internalword emerges in internalforms and token_sen, so add all of them
                        pos_wordsintoken_sen += 1
                        l += 1
                    #append the '|'
                    taggedsen_byme.append(u'|')                
                    pos_wordsintoken_sen += 1
                    #append the parse of right part
                    r=0
                    while r < len_rightpart:
                        dispword = dispformstree_sen[pos_dispforms]
                        taggedsen_byme.append(dispword.string+'/'+dispword.type+'/'+dispword.lemma)
                        pos_dispforms += 1 #the dispword emerges in dispforms and token_sen, so add all of them
                        pos_wordsintoken_sen += 1
                        r += 1
                else:
                    print 'there is a unexpected process in here!'
                    print 'Albedo can affect the [[electrical energy]] output of solar [[photovoltaic system|photovoltaic device]]s.'
                    print 'the above example can be handeled normally! but if there is unbalanced brackets, may be error!'
                    print 8/0
                
                #append the ']]'
                taggedsen_byme.append(u']]')
                pos_wordsintoken_sen += 1            
                
                continue

    except:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        print '------------------error information--------------------' 
        print 'An error occurred on line {} in statement {}. MergeParseResultofOneSentence(token_sen, dispformstree_sen, internalformstree_sen)'.format(line, text)
        print 'filename:',
        print inputfile.filename()
        print 'token_sen:',
        print token_sen#Albedo can affect the [[ electrical energy ]] output of solar [[ photovoltaic system | photovoltaic device ]] .
        print 'dispformstree_sen:',
        print dispformstree_sen.string
        print dispformstree_sen
        print 'internalformstree_sen:',
        print internalformstree_sen.string
        print internalformstree_sen
        print 'taggedsen_byme:'
        print taggedsen_byme
        print 'word:%s  pos_dispforms:%d  pos_internalforms:%d  pos_wordsintoken_sen:%d' % (word,pos_dispforms,pos_internalforms,pos_wordsintoken_sen)
        print 3/0
             
             
     
     
    #print '-------------------'
    #print 'token_sen:',
    #print token_sen
    #print 'dispformstree_sen:',
    #print dispformstree_sen.string
    #print dispformstree_sen
    #print 'internalformstree_sen:',
    #print internalformstree_sen.string
    #print internalformstree_sen
    #print 'taggedsen_byeme:',
    #print taggedsen_byme
    return taggedsen_byme


def findpos(word, wordsintoken_sen, beginpos):
    '''
    word: target word
    wordsintoken_sen: the list of words
    beingpos: the begin pos to find word
    return the pos of target word, emerage first
    '''
    for pos in range(beginpos, len(wordsintoken_sen),1):
        w = wordsintoken_sen[pos]
        if w == word:
            return pos
    return None

def findMatchingBraces(text, ldelim=0):
    """
    luwpeng: copy from WikiExtractor_Lemmatize(originalversion).py

    :param ldelim: number of braces to match. 0 means match [[]], {{}} and {{{}}}.
    """
    # Parsing is done with respect to pairs of double braces {{..}} delimiting
    # a template, and pairs of triple braces {{{..}}} delimiting a tplarg.
    # If double opening braces are followed by triple closing braces or
    # conversely, this is taken as delimiting a template, with one left-over
    # brace outside it, taken as plain text. For any pattern of braces this
    # defines a set of templates and tplargs such that any two are either
    # separate or nested (not overlapping).

    # Unmatched double rectangular closing brackets can be in a template or
    # tplarg, but unmatched double rectangular opening brackets cannot.
    # Unmatched double or triple closing braces inside a pair of
    # double rectangular brackets are treated as plain text.
    # Other formulation: in ambiguity between template or tplarg on one hand,
    # and a link on the other hand, the structure with the rightmost opening
    # takes precedence, even if this is the opening of a link without any
    # closing, so not producing an actual link.

    # In the case of more than three opening braces the last three are assumed
    # to belong to a tplarg, unless there is no matching triple of closing
    # braces, in which case the last two opening braces are are assumed to
    # belong to a template.

    # We must skip individual { like in:
    #   {{#ifeq: {{padleft:|1|}} | { | | &nbsp;}}
    # We must resolve ambiguities like this:
    #   {{{{ }}}} -> { {{{ }}} }
    #   {{{{{ }}}}} -> {{ {{{ }}} }}
    #   {{#if:{{{{{#if:{{{nominee|}}}|nominee|candidate}}|}}}|...}}
    #   {{{!}} {{!}}}

    # Handle:
    #   {{{{{|safesubst:}}}#Invoke:String|replace|{{{1|{{{{{|safesubst:}}}PAGENAME}}}}}|%s+%([^%(]-%)$||plain=false}}
    # as well as expressions with stray }:
    #   {{{link|{{ucfirst:{{{1}}}}}} interchange}}}

    if ldelim:  # 2-3
        reOpen = re.compile('[{]{%d,}' % ldelim)  # at least ldelim
        reNext = re.compile('[{]{2,}|}{2,}')  # at least 2
    else:
        reOpen = re.compile('{{2,}|\[{2,}')
        reNext = re.compile('{{2,}|}{2,}|\[{2,}|]{2,}')  # at least 2

    cur = 0
    while True:
        m1 = reOpen.search(text, cur)
        if not m1:
            return
        lmatch = m1.end() - m1.start()
        if m1.group()[0] == '{':
            stack = [lmatch]  # stack of opening braces lengths
        else:
            stack = [-lmatch]  # negative means [
        end = m1.end()
        while True:
            m2 = reNext.search(text, end)
            if not m2:
                return  # unbalanced
            end = m2.end()
            brac = m2.group()[0]
            lmatch = m2.end() - m2.start()

            if brac == '{':
                stack.append(lmatch)
            elif brac == '}':
                while stack:
                    openCount = stack.pop()  # opening span
                    if openCount == 0:  # illegal unmatched [[
                        continue
                    if lmatch >= openCount:
                        lmatch -= openCount
                        if lmatch <= 1:  # either close or stray }
                            break
                    else:
                        # put back unmatched
                        stack.append(openCount - lmatch)
                        break
                if not stack:
                    yield m1.start(), end - lmatch
                    cur = end
                    break
                elif len(stack) == 1 and 0 < stack[0] < ldelim:
                    # ambiguous {{{{{ }}} }}
                    #yield m1.start() + stack[0], end
                    cur = end
                    break
            elif brac == '[':  # [[
                stack.append(-lmatch)
            else:  # ]]
                while stack and stack[-1] < 0:  # matching [[
                    openCount = -stack.pop()
                    if lmatch >= openCount:
                        lmatch -= openCount
                        if lmatch <= 1:  # either close or stray ]
                            break
                    else:
                        # put back unmatched (negative)
                        stack.append(lmatch - openCount)
                        break
                if not stack:
                    yield m1.start(), end - lmatch
                    cur = end
                    break
                # unmatched ]] are discarded
                cur = end

def RemoveNestWikiLink(oldline):
    '''
    abc1111File[[2222 Khaled2222]] adkd222[[333]]222King [[ra222]]11119876So
    if the oldline is as above,
    then, only reserve the most deep link, that is [[333]].
    we want to reduce the trouble in Parsing sentence.
    :param u'abc[[1111File[[2222 Khaled2222]] adkd[[222[[333]]222]]King [[ra222]]1111]]9876So'
    :return:abc1111File[[2222 Khaled2222]] adkd222[[333]]222King [[ra222]]11119876So

    :param [[File:Cheb Khaled performed in Oran on July 5th 2011.jpg|thumb|[[Khaled (musician)|Cheb Khaled]] King [[ra?]]]]
    :return:File:Cheb Khaled performed in Oran on July 5th 2011.jpg|thumb|[[Khaled (musician)|Cheb Khaled]] King [[ra?]]


    '''
    #oldline='111File:aaa|thumb|[[2222Khaled (musician)|Cheb Khaled2222]] adkd'

    listSquare=[]
    for begin,end in findMatchingBraces(oldline):
        if oldline[begin] == '[' and oldline[begin + 1] == '[' and oldline[end - 1] == ']' and oldline[end - 2] == ']':
            listSquare.append((begin,end))
    ##record the string that not belong to any [[]]
    strNoSquare = []
    beginold = 0
    endold = len(oldline)
    for (begin,end) in listSquare:
        t = oldline[beginold:begin]
        beginold = end
        strNoSquare.append(t)
    t = oldline[beginold:endold]
    strNoSquare.append(t)
    #for s in strNoSquare:
    #    print s

    ##
    strSquare=[]
    for (begin,end) in listSquare:
        part = oldline[begin:end]
        partremoveedge = part[2:-2]
        beginsquare = partremoveedge.find('[[')
        endsquare = partremoveedge.find(']]')
        if beginsquare != -1 and endsquare != -1 and beginsquare < endsquare:  # if there are deeper [[]]
            t=RemoveNestWikiLink(partremoveedge)
            strSquare.append(t)
        else:  # if there is not deeper [[]]
            t = part
            strSquare.append(t)


    ##merge
    if len(strSquare)==0:
        #print ''.join(strNoSquare)
        return ''.join(strNoSquare)
    else:
        assert len(strNoSquare) - len(strSquare) == 1
        mergelist =[]
        for i in range(0,len(strNoSquare)):
            mergelist.append(strNoSquare[i])
            if i < len(strSquare):
                mergelist.append(strSquare[i])
        #print ''.join(mergelist)
        return ''.join(mergelist)

def ParseOneLine(oldline, inputfile):

    '''
    for a line from WikiExtractor with --html --lustyle, output its parsing result
    such as,
    input: Woodcut from a [[Diggers]] document by [[William Everard (Digger)|William Everard]] 
    output: Woodcut/NNP/woodcut from/IN/from a/DT/a [[ Diggers/NNP/diggers ]] document/NN/document by/IN/by [[ William/NNP/william Everard/NNP/everard (/(/( Digger/NNP/digger )/)/) | William/NNP/william Everard/NNP/everard ]]
    '''
    #oldline = 'External links.'
    #oldline = 'Albedo can affect the [[electrical energy]] output of solar [[photovoltaic system|photovoltaic device]]s.'
    #oldline = 'Albedo [[photovoltaic system|photovoltaic device]].'
    #oldline = 'test [[error [[photovoltaic system|photovoltaic device]] | errot]].'
    #oldline = '[[electrical energy]]'
    #oldline = 'Albedo can affect the [[electrical energy]] output  of solar [[photovoltaic system|photovoltaic device]]s.'
    #oldline = '[[electrical energy]] of | [[solar [[photovoltaic system|photovoltaic device]]s. dfsd]] [[asdf|e\yuio]]'
    #oldline = '<h3>Russian Revolution and other uprisings of the 1910s</h3>'    
    #oldline = 'According to the 2011 census conducted by [[Statistics Mauritius]], [[Hinduism]] is the major religion at 48.5%, followed by [[Christianity]] (32.7%), [[Islam]] (17.3%) and [[Buddhism]] (0.4%). Those of other religions accounted for 0.2% of the population, while [[irreligiosity|non-religious]] individuals were 0.7%. Finally, 0.1% refused to fill in any data. Mauritius is the only country in [[Africa]] to have a [[Hindu]] plurality.'
    #oldline = 'the [[Soviet Union|Soviet]]-allied party actively resisted attempts at [[collectivisation]] enactment.'
    #oldline = '" [[Spanish Communist Party]]-led [[Workers\' Party of Marxist Unification|dissident Marxists]] and anarchists.'    
    #oldline = 'According to Lovecraft\'s "History of the Necronomicon" (written 1927, first published [[1938 in literature|1938]]), Alhazred was:'
    #oldline = '<doc id="4729" url="https://en.wikipedia.org/wiki?curid=4729" title="Batman &amp; Robin (film)">'
    #oldline = 'to buy a car is parsed like <nowiki>to [buy [a car]]</nowiki>, rather not like <nowiki>[to buy] [a car]</nowiki>.'
    #oldline = 'abc[[111File:aaa|thumb|[[2222Khaled (musician)|Cheb Khaled2222]] adkd[[222[[333]]222]]]]King [[ra222]]1111]]9876So'# a house is a being, a persons body is a being, a tree is a being, a particular]]s, or [[concrete]] things, or [[matter]], or maybe [[Substance theory|substance]]s (but bear in mind the word substance has some special philosophical meanings). '
    #oldline = 'abc[[1111File[[2222 Khaled2222]] adkd[[222[[333]]222]]King [[ra222]]1111]]9876So'
    #oldline = '[[File:Cheb Khaled performed in Oran on July 5th 2011.jpg|thumb|[[Khaled (musician)|Cheb Khaled]] King [[ra?]]]]'
    #oldline = 'If a point on a two-dimensional plane is , then the [[Distance from a point to a line|distance]] of the point from the x axis is [[Absolute value (algebra)|<nowiki>|</nowiki>y<nowiki>|</nowiki>]] and the distance of the point from the y axis is |x|.'
    #oldline = '[[File:AntonioNegri SeminarioInternacionalMundo.jpg|upright|thumb|[[Antonio Negri]], main theorist of Italian autonomism]]Autonomism refers '
    #oldline ='e.g. "<nowiki>[x [is y]]" or "[x [does y]]</nowiki>". Recently, this model of semantics has been complemente'
    #oldline = 'to buy a car is parsed like <nowiki>to [buy [a car]]</nowiki>, rather not like <nowiki>[to buy] [a car]</nowiki>.'
    #oldline = '(as in "<nowiki>[[the chimpanzee]\'s lips]</nowiki>") or a clause can contain another clause (as in "<nowiki>[I see [the dog is running]]</nowiki>")'
    #oldline = 'So a house is a being, a person\'s body is a being, a tree is a being, a particular]]s, or [[concrete]] things, or [[matter]], or maybe [[Substance theory|substance]]s (but bear in mind the word \'substance\' has some special philosophical meanings). '
    #oldline = 'abc[[1111File[[2222 Khaled2222]] adkd[[222[[3[[44]]33]]222]]King [[ra2[[333]]22]]1111]]9876So'
    #oldline = '[[Category:Music video networks in Australia|Channel [V]]] '

    #print oldline
    oldline = oldline.replace(r'\n', r'')  #2017-03-30
    ### the speical character need to be replaced 2017-03-28
    oldline = oldline.replace('&quot;', '"')
    oldline = oldline.replace('&#039;', '\'')
    oldline = oldline.replace('&amp;', '&')
    ###
    ###process <nowiki> tag 2017-03-30
    tcount = 0
    while 1:
        tcount += 1
        if tcount >= 300:
            print 'there may be a serve error for <nowiki> ! You must check it!'
            print 'inputfile:',
            print inputfile
            print 'oldline:',
            print oldline
            print 3 / 0
        pos_beginnowiki = oldline.find('<nowiki>')
        pos_endnowiki = oldline.find('</nowiki>')
        if pos_beginnowiki==-1 and pos_endnowiki==-1:
            break
        elif pos_beginnowiki!=-1 and pos_endnowiki!=-1 and pos_beginnowiki<pos_endnowiki:
            nowiki = ReTag_nowiki.search(oldline)
            if nowiki:
                conold = nowiki.group(1)
                connew = conold.replace('[', '')
                connew = connew.replace(']', '')
                old = '<nowiki>' + conold + '</nowiki>'
                oldline = oldline.replace(old, connew)
        elif pos_beginnowiki==-1 and pos_endnowiki!=-1:
            oldline = oldline.replace('</nowiki>','',1)# replace only once
        elif pos_beginnowiki!=-1 and pos_endnowiki==-1:
            oldline = oldline.replace('<nowiki>','',1)# replace only once
        elif pos_beginnowiki!=-1 and pos_endnowiki!=-1 and pos_beginnowiki>pos_endnowiki:
            oldline = oldline.replace('</nowiki>','',1)# replace only once
        else:
            print 'there may be a serve error  for <nowiki> ! You must check it!'
            print 'inputfile:',
            print inputfile
            print 'oldline:',
            print oldline
            print 3 / 0


    ###remove the nest brackets 2017-03-30
    #print oldline
    oldline = RemoveNestWikiLink(oldline)
    #print oldline



    sectiontitlematch = ReTag_sectiontitle.match(oldline)
    if sectiontitlematch:
        num = sectiontitlematch.group(1)
        text = sectiontitlematch.group(2)
        return '<h%s> %s </h%s>' % (num, ParseOneLine(text, inputfile), num)
    
    doctitlematch = ReTag_doctitle.match(oldline)
    if doctitlematch:
        #return oldline
        return re.sub(r'\n',r'',oldline)
        
    categorylinematch = ReTag_categoryline.match(oldline)
    if categorylinematch:
        category = categorylinematch.group(1)
        tpos = oldline.find(u':[[Category:')
        if tpos>1: #[[Category:[[Category:Conflicts in 881]]
            oldline = oldline[tpos+1:]
            categorylinematch = ReTag_categoryline.match(oldline)
            if categorylinematch:
                category = categorylinematch.group(1)
        if category.find(u'[')!=-1 or category.find(u']')!=-1:#[[Category:Music video networks in Australia|Channel [V]]]
            return u''
        else:
            return re.sub(r'\n',r'',oldline)
        

    #revise original line to avoid some furthur errors
    line = ReTag_wikilinkhyphen.sub(r']] [[',oldline)#this line should be former of ReTag_wikilinkpostfix. [[neurexin]]–[[neuroligin]]     
    
    line = ReTag_wikilinkpostfix.sub(r'[[\1]]',line)#[[  ]] de qianhouzhui
    line = ReTag_wikilinkprefix.sub(r'[[\2]]',line)

    line = ReTag_wikilinkpunctuationfix1.sub(r'[[\1]]',line)# [[]] zhong cun zai yinqi duanju cuowu de fuhao
    line = ReTag_wikilinkpunctuationfix2.sub(r'[[\2|\3]]',line)    

    line = ReTag_wikilinksegsentence.sub(r'\1 \2',line)# [[desegregation]].The civil   ---->   [[desegregation]]. The civil
    line = reTag_wikilinkpresymbol.sub(r'\1 \2',line)#e Electron include:[[Crystal Castles  -----> e Electron include: [[Crystal Castles 
                                                     #:[[18th Flight Test Squadron]]  ---->  : [[18th Flight Test Squadron]]

    line = re.sub(r': \[\[',r', [[',line)# there is a bug for ': [[' in pattern.en.tokenize()  'London: [[Pluto Press]], 2007'
    
    
    
    line = ReTag_blankwikilink.sub(r' ',line)
    line = ReTag_x85etc.sub(r' ',line)
    line = ReTag_space.sub(r' ',line)


    

    #print originalline

    #print 'change original line:'
    originalline = ReTag_wikilink1.sub(r'[[ \1 ]]',line)
    originalline = ReTag_wikilink2.sub(r'[[ \2 | \3 ]]',originalline)#avoid 'devices' can't be find in next process
    #print originalline       
    
    #change from here for multiple sentences in a line
    token_sens = tokenize(originalline)#Albedo can affect the [ [ electrical energy ] ] output of lusolar [ [ photovoltaic system|photovoltaic device ] ] .
    
    ret = []    
    id_sen = 0
    while id_sen < len(token_sens):
        #print '------------------------------'
        #print 'the %d-th sentence:' %(id_sen)
        
        token_sen = token_sens[id_sen]
        token_sen = ReTag_tokenizebug1.sub(r':- [ [',token_sen)#fix the little bug of tokenize
        token_sen = ReTag_tokenizebug2.sub(r': [ [',token_sen)
        token_sen = ReTag_wikilink1_r.sub(r'[[ \1 ]]', token_sen)
        token_sen = ReTag_wikilink2_r.sub(r'[[ \2 | \3 ]]', token_sen)#[[ Presbyterian Church in America | PCA ]] , 009 members in 108 congregations , [[ PC(USA ) ]]
        
        tmp = ReTag_wikilink1_aftertoken.sub(r'\1',token_sen) #[[electrical energy]]   --> electrical energy
        dispforms_sen = ReTag_wikilink2_aftertoken.sub(r'\3',tmp)#[[photovoltaic system|photovoltaic device]]s  --> photovoltaic device
        
        tmp = ReTag_wikilink1_aftertoken.sub(r'\1',token_sen) #[[electrical energy]]   --> electrical energy
        internalforms_sen = ReTag_wikilink2_aftertoken.sub(r'\2', tmp) #[[photovoltaic system|photovoltaic device]]s  --> photovoltaic system
        
        dispformstree = parsetree(dispforms_sen, tokenize = False, tags = True, chunks = False, lemmata = True)
        if dispforms_sen == internalforms_sen:
            internalformstree = dispformstree
        else:
            internalformstree = parsetree(internalforms_sen, tokenize = False, tags = True, chunks = False, lemmata = True)
            
        try:
            assert len(dispformstree) == 1 and len(internalformstree) == 1,\
            "the length of dispformstree or internalformstree isn't 1. this is not right!"
        except AssertionError, reason:
            print 'AssertionError: the length of dispformstree and internalformstree are not 1 ! They are %d and %d seperately' % (len(dispformstree), len(internalformstree))
            print "%s:%s" % (reason.__class__.__name__, reason)
            print 'filename:',
            print inputfile.filename()
            print 'oldline:',
            print oldline.encode('utf-8')
            print 'token_sens:    len:%d' % (len(token_sens))
            print token_sens
            print 'token_sen:',
            print token_sen
            print 'dispforms_sen:',
            print dispforms_sen
            print 'dispformstree:',
            print dispformstree
            print 'internalforms_sen:',
            print internalforms_sen
            print 'internalformstree',
            print internalformstree
            print 3/0
            
        dispformstree_sen = dispformstree[0]
        internalformstree_sen = internalformstree[0]

        cursenresult = MergeParseResultofOneSentence(token_sen, dispformstree_sen, internalformstree_sen, inputfile)
        ret.extend(cursenresult)
        #print ' '.join(cursenresult)
        
        id_sen += 1


    '''
    test = u'Category:[[Category:Conflicts/XX/Category:[[Category:Conflicts ' \
           u'<section/NN/<section begin="weather/NN/begin="weather box/NN/box "/"/" /><section/NN/ end="weather/DT/end="weather box/NN/box "/"/" />/NN/ ' \
           u'[[ [[/VBZ/[[ voiceless/JJ/voiceless dental/JJ/dental | [[/VBZ/[[ ]] ' \
           u'[[ //CC/ ]] [[ //CC/ ]] [[ //CC/ ]] ' \
           u'[[ //CC/ | x/x/x ]] [[ //CC/ | voiceless/JJ/voiceless dental/JJ/dental ]] [[ //CC/ | x/x/x ]] ' \
           u'[[ x/x/x | //CC/ ]] [[ voiceless/JJ/voiceless dental/JJ/dental | //CC/ ]] [[ x/x/x | //CC/ ]] ' \
           u'[[ //CC/ | //CC/ ]] [[ //CC/ | //CC/ ]] [[ //CC/ | //CC/ ]] ' \
           u'REF|[2]]][[#REF|[5/NNP/ref|[2]]][[#ref|[5 ]/)/] DEVS#References|[Giambiasi01]]][[DEVS#References|[Zacharewicz08/NNP/devs#references|[giambiasi01]]][[devs#references|[zacharewicz08 ' \
           u']/)/] 3,3)[2,2,2]]=[4,3,3/CD/3,3)[2,2,2]]=[4,3,3 ]/)/] ,/,/, view.jpg|thumb]]The/VBZ/view.jpg|thumb]]the journal)]],123/RB/journal)]],123 football]]er/NN/football]]er Carbon-14|[C]]]formate/NNP/carbon-14|[c]]]formate ' \
           u'1.1.1-propellane|[1.1.1]propellane]]s/CD/1.1.1-propellane|[1.1.1]propellane]]s ' \
           u'//CC/ [[ [[/VBZ/[[ voiceless/JJ/voiceless dental/JJ/dental | [[/VBZ/[[ ]]' \
           u'[[ | x/x/x ]] [[ x/x/x | ]] [[ | ]] [[ | x/x/x ]] [[ x/x/x | ]] [[ | ]] [[ | x/x/x ]] [[ x/x/x | ]] [[ | ]] [[ x/x/x ]] [[ x/y/z | x/y/z a/b/c ]]'
    '''
    #test = u'The/DT/the voiced/JJ/voiced implosives/NNS/implosive Voiced/VBN/voice bilabial/NN/bilabial [[ [/(/[ /NN/ ]/)/] | implosive/NN/implosive ]] |/NN/| [/(/[ /NN/ ]/)/] and/CC/and Voiced/JJ/voiced alveolar/NN/alveolar implosive/NN/implosive |/NN/| [/(/[ /NN/ ]/)/] may/MD/may have/VB/have contrasted/VBN/contrast with/IN/with [/(/[ ]/)/] and/CC/and [/(/[ ]/)/] ,/,/, which/WDT/which is/VBZ/be rather/RB/rather rare/JJ/rare ,/,/, or/CC/or the/DT/the two/CD/two sets/NNS/set may/MD/may have/VB/have The/DT/the vowels/NNS/vowel in/IN/in parentheses/NNS/parenthesis are/VBP/be assumed/VBN/assume to/TO/to have/VB/have been/VBN/be used/VBN/used in/IN/in early/JJ/early Middle/NNP/middle Khmer/NNP/khmer but/CC/but this/DT/this has/VBZ/have never/RB/never been/VBN/be proved/VBN/prove nor/CC/nor disproved/VBN/disprove ././. In/IN/in addition/NN/addition to/TO/to the/DT/the vowel/NN/vowel nuclei/NNS/nuclei listed/VBD/list ,/,/, there/EX/there were/VBD/be two/CD/two diphthongs/NNS/diphthong inherited/VBD/inherit from/IN/from Old/NNP/old Khmer/NNP/khmer ,/,/, [/(/[ iə/NN/iə ]/)/] and/CC/and [/(/[ uə/NN/uə ]/)/] ,/,/, and/CC/and a/DT/a third/JJ/third ,/,/, [/(/[ ɨə/NN/ɨə ]/)/] ,/,/, entered/VBD/enter the/DT/the language/NN/language via/IN/via loanwords/NNS/loanword from/IN/from Thai/NNP/thai ././.'
    #ret = test.split(u' ')
    
    #print test
    #print test
    retstr = checkRetOfParseOneLine(ret)
    #print '#######'
    #print 'oldline:',
    #print oldline
    #print 'line:',
    #print line
    #print 'originalline:',
    #print originalline
    #print 'return:',
    #print ' '.join(ret)
    #return ret
    #ret.append('\n')

    #return ' '.join(ret)
    ret = None
    return retstr



def checkRetOfParseOneLine(originalret):
    #2017 - 6 - 6
    #在执行SSA2程序时，发现处理中一些错误，很多文件中止了。我仔细看了一下，原因是我最原始的ParseWikiExtractorOut.py程序就有问题。
    # 我这里对原来的结果中一些不规范的情况进行过滤处理。

    if originalret == None or len(originalret)==0:
        return u''
    remodifyret = []

    #逐个检查单个元素是不是合理
    for item in originalret:
        #modify   'le/VBP/le Carré]]-style/JJ/carré]]-style spy/NN/spy'
        m1 = ReTag_checkret_2rightbracket.search(item)
        if m1:
            token = m1.group(1)
            pos = m1.group(2)
            lemma = m1.group(3)
            item = u'{}/{}/{}'.format(token, pos, lemma)
        #modify  'le/NN/le Carré|[John/NNP/carré|[john ]/)/]'
        m2 = ReTag_checkret_vertlineleftbracket.search(item)
        if m2:
            token = m2.group(1)
            pos = m2.group(2)
            lemma = m2.group(3)
            item = u'{}/{}/{}'.format(token, pos, lemma)
        #remove '/><section/NN/' [/(/[ /NN/ ]/)/] |/NN/|
        if ReTag_checkret_beginendwithslash.search(item) and item!=u'[[' and item!=u']]':
            continue
        #remove '//CC/'
        if ReTag_checkret_3blackslash.search(item):
            continue
        #remove '[[/VB/[['
        m3 = ReTag_checkret_2leftbracket.search(item)
        if m3:
            #pos = m3.group(1)
            #item = u'{}/{}/{}'.format(u'x[x', pos, u'x[x')
            continue
        #remove the ietms that still contain ']]' or '[['
        if ReTag_checkret_2bracket.search(item) and item!=u'[[' and item!=u']]':
            continue
        #remove the items with more than 2 '/' 这个要求比较高了，损失可能有点大了，但是丢了也无所谓。反正这些反斜线多的，也通常不是概念concept
        slashmatch = ReTag_checkret_morethan2slash.findall(item)
        if len(slashmatch)!=0 and len(slashmatch)!=2:#<h2>Notable Alumni</sub></h2>  ->  <h2>Notable/JJ/<h2>notable Alumni</sub></h2>/NNP/alumni<
            continue
        #remove the item with [ or ] 这个要求也比较高了，除了独立的 [[ 与 ]] ，只要item中出现[或]，就全认为质量不高的，直接删掉了。就是为了减少后期更进一步处理时的麻烦
        if item!=u'[[' and item!=u']]' and (item.find(u'[')!=-1 or item.find(u']')!=-1):
            continue

        remodifyret.append(item)

    #检查上一步处理的结果是不是仍有一些意外
    newline = u' '.join(remodifyret)
    #print newline
    newline = ReTag_checkret_21.sub(u' ',newline)# [[ | x/x/x ]]  替换为1个空格
    newline = ReTag_checkret_22.sub(u' ',newline) #  [[ x/x/x | ]]  替换为1个空格
    newline = ReTag_checkret_23.sub(u' ',newline)#  [[ | ]]  替换为1个空格
    newline = ReTag_checkret_24.sub(u' ',newline) # [[ ]]
    newline = ReTag_checkret_25.sub(u' ',newline)#将多个连续空格替换为1个
    newline = newline.strip()
    #print newline


    originalret=None
    remodifyret=None
    return newline







def main():
    print 'this is main function'
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    parser.add_argument("input",
                        help="the Output Directory of WikiExtractor with --Lustyle")
                        
    groupO = parser.add_argument_group('Output')
    groupO.add_argument("-o", "--output", default="text",
                        help="directory for Parsed files (or '-' for dumping to stdout)")
    groupO.add_argument("-c", "--compress", action="store_true",
                        help="compress output files using bzip")
                        
    default_process_count = cpu_count() - 1
    parser.add_argument("--processes", type=int, default=default_process_count,
                        help="Number of processes to use (default %(default)s)")
                        
    groupS = parser.add_argument_group('Special')
    groupS.add_argument("-q", "--quiet", action="store_true",
                        help="suppress reporting progress info")
    groupS.add_argument("--debug", action="store_true",
                        help="print debug info")
    groupS.add_argument("-f", "--filesingle", action="store_true",
                        help="analyze a single file outputted by WikiExtractor(debug option)")
    groupS.add_argument("-v", "--version", action="version",
                        version='%(prog)s ' + version,
                        help="print program version")
                        
    args = parser.parse_args()    

    FORMAT = '%(levelname)s: %(message)s'
    logging.basicConfig(format=FORMAT)

    logger = logging.getLogger()
    if not args.quiet:
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    input_directory = args.input
    output_directory = args.output


    if args.filesingle:
        logger.info('Enter Single File mode ...')
        input_file = input_directory
        output_file = output_directory
        if os.path.isfile(input_file)==False:
            logger.error('Single file mode, But the input is not a file!')
            return
        Parse(input_file, output_file)
        #inputfile = fileinput.FileInput(input_file, openhook=fileinput.hook_compressed)
        #if output_file == 'text': #this means default value of '--output'
        #    outputfile = sys.stdout
        #else:
        #    outputfile = open(output_file,'w')
        
        #Parse(inputfile, outputfile)

        #inputfile.close()
        #outputfile.close()
        return
    else:
        if os.path.isdir(input_directory) == False:
            logger.error('Input is a non-existent directory')
            return
        if input_directory == output_directory:
            logger.error('Output directory should be different with input directory')
            return        


    if output_directory != '-' and not os.path.isdir(output_directory):
        try:
            os.makedirs(output_directory)
        except:
            logging.error('Could not create: %s', output_directory)
            return
    
    process_dump(input_directory, output_directory, args.compress, args.processes)
    #write a function(), get the name of input file and make a name corresponding with it as the 
            #name of output file
            #see reduce_process()

    #process_dump(input_file, args.templates, output_path, file_size,
    #             args.compress, args.processes)
                        
                        
def process_dump(input_directory, output_directory, file_compress, process_count):
    """
    :param input_directory: the input directory of ParseWikiExtracorOut, that is, the output directory of WikiExtractor, which includes: AA/wiki_00,wiki_01,....
    :param output_directory: directory where to store the parsed data
    :param file_compress: whether to compress files with bzip.
    :param process_count: number of extraction processes to spawn.
    """
    
    input_dirs = os.walk(input_directory) 
    input_files = [] #the files that need to be parsed
    for root, dirs, files in input_dirs: 
        for d in dirs:
            print 'child directory:',
            print os.path.join(root, d)   
            assert len(dirs) == 0, "input_directory: '%s' includes the above child directory! This should be avoided!" % input_directory
            print "input_directory: '%s' includes child directory: '%s' ! please notice this and preprocess them!" % (input_directory, os.path.join(root, d))          
        for f in files: 
            #print os.path.join(root, f) 
            input_files.append(os.path.join(root, f))
            
    
    if os.path.exists(output_directory) == False:
        os.mkdir(output_directory)
    else:
        output_dirs = os.walk(output_directory)
        for root, dirs, files in output_dirs:
            if len(dirs)!=0 or len(files)!=0:
                print 'output directory:',
                print output_directory
                assert len(dirs)==0 and len(files)==0, "Please clean up the output directroy! "
            

    # process pages
    logging.info("Starting to parse the files outputted by WikiExtractor from %s.", input_directory)
    parse_start = default_timer()

    # Parallel:
    # - Files to be processed are dispatched to workers

    maxsize = 1 * process_count
    
    worker_count = max(1, process_count)

    # initialize jobs queue
    jobs_queue = Queue(maxsize=maxsize)

    # start worker processes
    logging.info("Using %d parser processes.", worker_count)
    workers = []
    for i in range(worker_count):
        parser = Process(target=parse_process,
                            args=(i, jobs_queue))
        parser.daemon = True  # only live while parent process lives
        parser.start()
        workers.append(parser)

    file_num = 0
    # Mapper process
    for infile in input_files:
        basename = os.path.basename(infile)
        outfile = output_directory + basename
        #print infile
        #print outfile
        job = (infile, outfile)
        jobs_queue.put(job)
        file_num += 1
        
    logging.info("There are %d files that need to be parsed in the input directory", file_num)
        


    # signal termination
    for _ in workers:
        jobs_queue.put(None)
    # wait for workers to terminate
    for w in workers:
        w.join()


    parse_duration = default_timer() - parse_start
    parse_average_time = parse_duration / file_num
    logging.info("Finished %d-process parse of %d files in %.1fs (average time: %.1f s/file)",
                 process_count, file_num, parse_duration, parse_average_time)    
    

# Multiprocess support
def parse_process(i, jobs_queue):
    """Pull tuples of raw page content, do CPU/regex-heavy fixup, push finished text
    :param i: process id.
    :param jobs_queue: where to get jobs.
    """
    logging.info("enter %d-th parse_process pid:%d",i,os.getpid());
    while True:
        logging.info("parse_process pid:%d 'while True' try to get a job from jobs_queue. current jobs_queue's size:%d", os.getpid(), jobs_queue.qsize())
        job = jobs_queue.get()  # job is (id, title, page, page_num)
                                #lu : jobs_queue.get() would block current extract process, until it can return a job object.
        if job:
            infile, outfile = job
            logging.info("parse_process pid:%d get a job(infile:%s outfile:%s) from jobs_queue to parse... jobs_queue's size become:%d", os.getpid(), infile, outfile, jobs_queue.qsize())
            try:
                Parse(infile, outfile)
            except:
                logging.exception('parse_process pid:%d Occur a expect, when Parsing File: %s', os.getpid(), infile)
                
            logging.info("parse_process pid:%d the job(infile:%s outfile:%s) has been parsed, whose size become:%d", os.getpid(), infile, outfile, jobs_queue.qsize())
        else:
            logging.info("parse_process pid:%d the job gotten from job_queue is 'None', so to quit...",os.getpid())
            logging.debug('Quit extractor')
            break





if __name__ == '__main__':
    main()