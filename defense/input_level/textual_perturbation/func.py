"""
Chew, O., Lu, P. Y., Lin, J., & Lin, H. T.
Defending Text-to-image Diffusion Models: Surprising Efficacy of Textual Perturbations Against Backdoor Attacks.
In ECCV 2024 Workshop The Dark Side of Generative AIs and Beyond.
Source: https://github.com/oscarchew/t2i-backdoor-defense
"""

from textattack.transformations import WordSwapRandomCharacterDeletion
from textattack.transformations import WordSwapQWERTY
from textattack.transformations import WordSwapEmbedding
from textattack.transformations import CompositeTransformation
from textattack.transformations.sentence_transformations import BackTranslation
from textattack.constraints.pre_transformation import RepeatModification
from textattack.constraints.pre_transformation import StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.augmentation import Augmenter
from textattack.shared.utils import set_seed
from textattack.shared import AttackedText
import random
import json

# raw data for generating reverse homoglyph dictionary
# data from https://github.com/codebox/homoglyph/blob/master/raw_data/char_codes.txt
homoglyph_mapping = {
    'o': [
        '\u0030', '\u039f', '\u03bf', '\u03c3', '\u041e', '\u043e', '\u0555', '\u0585', '\u05e1',
        '\u0647', '\u0665', '\u06be', '\u06c1', '\u06d5', '\u06f5', '\u07c0', '\u0966', '\u09e6', '\u0a66', '\u0ae6',
        '\u0b20', '\u0b66', '\u0be6', '\u0c02', '\u0c66', '\u0c82', '\u0ce6', '\u0d02', '\u0d20', '\u0d66', '\u0d82',
        '\u0e50', '\u0ed0', '\u101d', '\u1040', '\u10ff', '\u12d0', '\u1d0f', '\u1d11', '\u2134', '\u2c9e', '\u2c9f',
        '\u2d54', '\u3007', '\ua4f3', '\uab3d', '\ufba6', '\ufba7', '\ufba8', '\ufba9', '\ufbaa', '\ufbab', '\ufbac',
        '\ufbad', '\ufee9', '\ufeea', '\ufeeb', '\ufeec', '\uff10', '\uff2f', '\uff4f', '\u10292', '\u102ab', '\u10404',
        '\u1042c', '\u104c2', '\u104ea', '\u10516', '\u114d0', '\u118b5', '\u118c8', '\u118d7', '\u118e0', '\u1d40e',
        '\u1d428', '\u1d442', '\u1d45c', '\u1d476', '\u1d490', '\u1d4aa', '\u1d4de', '\u1d4f8', '\u1d512', '\u1d52c',
        '\u1d546', '\u1d560', '\u1d57a', '\u1d594', '\u1d5ae', '\u1d5c8', '\u1d5e2', '\u1d5fc', '\u1d616', '\u1d630',
        '\u1d64a', '\u1d664', '\u1d67e', '\u1d698', '\u1d6b6', '\u1d6d0', '\u1d6d4', '\u1d6f0', '\u1d70a', '\u1d70e',
        '\u1d72a', '\u1d744', '\u1d748', '\u1d764', '\u1d77e', '\u1d782', '\u1d79e', '\u1d7b8', '\u1d7bc', '\u1d7ce',
        '\u1d7d8', '\u1d7e2', '\u1d7ec', '\u1d7f6', '\u1ee24', '\u1ee64', '\u1ee84', '\u1fbf0'
    ],
    'i': [
        '\u0031', '\u007c', '\u0131', '\u0196', '\u01c0', '\u0269', '\u026a', '\u02db',
        '\u037a', '\u0399', '\u03b9', '\u0406', '\u0456', '\u04c0', '\u04cf', '\u05c0', '\u05d5', '\u05df', '\u0627',
        '\u0661', '\u06f1', '\u07ca', '\u13a5', '\u16c1', '\u1fbe', '\u2110', '\u2111', '\u2113', '\u2139', '\u2148',
        '\u2160', '\u2170', '\u217c', '\u2223', '\u2373', '\u23fd', '\u2c92', '\u2d4f', '\ua4f2', '\ua647', '\uab75',
        '\ufe8d', '\ufe8e', '\uff11', '\uff29', '\uff49', '\uff4c', '\uffe8', '\u1028a', '\u10309', '\u10320', '\u118c3',
        '\u16f28', '\u1d408', '\u1d422', '\u1d425', '\u1d43c', '\u1d456', '\u1d459', '\u1d470', '\u1d48a', '\u1d48d',
        '\u1d4be', '\u1d4c1', '\u1d4d8', '\u1d4f2', '\u1d4f5', '\u1d526', '\u1d529', '\u1d540', '\u1d55a', '\u1d55d',
        '\u1d574', '\u1d58e', '\u1d591', '\u1d5a8', '\u1d5c2', '\u1d5c5', '\u1d5dc', '\u1d5f6', '\u1d5f9', '\u1d610',
        '\u1d62a', '\u1d62d', '\u1d644', '\u1d65e', '\u1d661', '\u1d678', '\u1d692', '\u1d695', '\u1d6a4', '\u1d6b0',
        '\u1d6ca', '\u1d6ea', '\u1d704', '\u1d724', '\u1d73e', '\u1d75e', '\u1d778', '\u1d798', '\u1d7b2', '\u1d7cf',
        '\u1d7d9', '\u1d7e3', '\u1d7ed', '\u1d7f7', '\u1e8c7', '\u1ee00', '\u1ee80', '\u1fbf1'
    ],
    'A': [
        '\u0391', '\u0410', '\u13aa', '\u15c5', '\u1d00', '\ua4ee', '\uab7a', '\uff21', '\u102a0', '\u16f40',
        '\u1d400', '\u1d434', '\u1d468', '\u1d49c', '\u1d4d0', '\u1d504', '\u1d538', '\u1d56c', '\u1d5a0', '\u1d5d4',
        '\u1d608', '\u1d63c', '\u1d670', '\u1d6a8', '\u1d6e2', '\u1d71c', '\u1d756', '\u1d790'
    ],
    'B': [
        '\u0299', '\u0392', '\u0412', '\u0432', '\u13f4', '\u13fc', '\u15f7', '\u16d2', '\u212c', '\ua4d0',
        '\ua7b4', '\uff22', '\u10282', '\u102a1', '\u10301', '\u1d401', '\u1d435', '\u1d469', '\u1d4d1', '\u1d505',
        '\u1d539', '\u1d56d', '\u1d5a1', '\u1d5d5', '\u1d609', '\u1d63d', '\u1d671', '\u1d6a9', '\u1d6e3', '\u1d71d',
        '\u1d757', '\u1d791'
    ],
    'C': [
        '\u03f9', '\u0421', '\u13df', '\u1455', '\u2102', '\u212d', '\u216d', '\u2282', '\u2ca4', '\u2e26',
        '\ua4da', '\uff23', '\u102a2', '\u10302', '\u10415', '\u1051c', '\u118e9', '\u118f2', '\u1d402', '\u1d436',
        '\u1d46a', '\u1d49e', '\u1d4d2', '\u1d56e', '\u1d5a2', '\u1d5d6', '\u1d60a', '\u1d63e', '\u1d672', '\u1f74c'
    ],
    'D': [
        '\u13a0', '\u15de', '\u15ea', '\u1d05', '\u2145', '\u216e', '\ua4d3', '\uab70', '\uff24', '\u1d403',
        '\u1d437', '\u1d46b', '\u1d49f', '\u1d4d3', '\u1d507', '\u1d53b', '\u1d56f', '\u1d5a3', '\u1d5d7', '\u1d60b',
        '\u1d63f', '\u1d673'
    ],
    'E': [
        '\u0395', '\u0415', '\u13ac', '\u1d07', '\u2130', '\u22ff', '\u2d39', '\ua4f0', '\uab7c', '\uff25',
        '\u10286', '\u118a6', '\u118ae', '\u1d404', '\u1d438', '\u1d46c', '\u1d4d4', '\u1d508', '\u1d53c', '\u1d570',
        '\u1d5a4', '\u1d5d8', '\u1d60c', '\u1d640', '\u1d674', '\u1d6ac', '\u1d6e6', '\u1d720', '\u1d75a', '\u1d794'
    ],
    'F': [
        '\u03dc', '\u15b4', '\u2131', '\ua4dd', '\ua798', '\uff26', '\u10287', '\u102a5', '\u10525',
        '\u118a2', '\u118c2', '\u1d213', '\u1d405', '\u1d439', '\u1d46d', '\u1d4d5', '\u1d509', '\u1d53d', '\u1d571',
        '\u1d5a5', '\u1d5d9', '\u1d60d', '\u1d641', '\u1d675', '\u1d7ca'
    ],
    'G': [
        '\u0262', '\u050c', '\u050d', '\u13c0', '\u13f3', '\u13fb', '\ua4d6', '\uab90', '\uff27', '\u1d406',
        '\u1d43a', '\u1d46e', '\u1d4a2', '\u1d4d6', '\u1d50a', '\u1d53e', '\u1d572', '\u1d5a6', '\u1d5da', '\u1d60e',
        '\u1d642', '\u1d676'
    ],
    'H': [
        '\u029c', '\u0397', '\u041d', '\u043d', '\u13bb', '\u157c', '\u210b', '\u210c', '\u210d', '\u2c8e',
        '\ua4e7', '\uab8b', '\uff28', '\u102cf', '\u1d407', '\u1d43b', '\u1d46f', '\u1d4d7', '\u1d573', '\u1d5a7',
        '\u1d5db', '\u1d60f', '\u1d643', '\u1d677', '\u1d6ae', '\u1d6e8', '\u1d722', '\u1d75c', '\u1d796'
    ],
    'J': [
        '\u037f', '\u0408', '\u13ab', '\u148d', '\u1d0a', '\ua4d9', '\ua7b2', '\uab7b', '\uff2a', '\u1d409',
        '\u1d43d', '\u1d471', '\u1d4a5', '\u1d4d9', '\u1d50d', '\u1d541', '\u1d575', '\u1d5a9', '\u1d5dd', '\u1d611',
        '\u1d645', '\u1d679'],
    'K': [
        '\u039a', '\u041a', '\u13e6', '\u16d5', '\u212a', '\u2c94', '\ua4d7', '\uff2b', '\u10518', '\u1d40a',
        '\u1d43e', '\u1d472', '\u1d4a6', '\u1d4da', '\u1d50e', '\u1d542', '\u1d576', '\u1d5aa', '\u1d5de', '\u1d612',
        '\u1d646', '\u1d67a', '\u1d6b1', '\u1d6eb', '\u1d725', '\u1d75f', '\u1d799'
    ],
    'L': [
        '\u029f', '\u13de', '\u14aa', '\u2112', '\u216c', '\u2cd0', '\u2cd1', '\ua4e1', '\uabae', '\uff2c',
        '\u1041b', '\u10443', '\u10526', '\u118a3', '\u118b2', '\u16f16', '\u1d22a', '\u1d40b', '\u1d43f', '\u1d473',
        '\u1d4db', '\u1d50f', '\u1d543', '\u1d577', '\u1d5ab', '\u1d5df', '\u1d613', '\u1d647', '\u1d67b'
    ],
    'M': [
        '\u039c', '\u03fa', '\u041c', '\u13b7', '\u15f0', '\u16d6', '\u2133', '\u216f', '\u2c98', '\ua4df',
        '\uff2d', '\u102b0', '\u10311', '\u1d40c', '\u1d440', '\u1d474', '\u1d4dc', '\u1d510', '\u1d544', '\u1d578',
        '\u1d5ac', '\u1d5e0', '\u1d614', '\u1d648', '\u1d67c', '\u1d6b3', '\u1d6ed', '\u1d727', '\u1d761', '\u1d79b'
    ],
    'N': [
        '\u0274', '\u039d', '\u2115', '\u2c9a', '\ua4e0', '\uff2e', '\u10513', '\u1d40d', '\u1d441',
        '\u1d475', '\u1d4a9', '\u1d4dd', '\u1d511', '\u1d579', '\u1d5ad', '\u1d5e1', '\u1d615', '\u1d649', '\u1d67d',
        '\u1d6b4', '\u1d6ee', '\u1d728', '\u1d762', '\u1d79c'
    ],
    'P': [
        '\u03a1', '\u0420', '\u13e2', '\u146d', '\u1d18', '\u1d29', '\u2119', '\u2ca2', '\ua4d1', '\uabb2',
        '\uff30', '\u10295', '\u1d40f', '\u1d443', '\u1d477', '\u1d4ab', '\u1d4df', '\u1d513', '\u1d57b', '\u1d5af',
        '\u1d5e3', '\u1d617', '\u1d64b', '\u1d67f', '\u1d6b8', '\u1d6f2', '\u1d72c', '\u1d766', '\u1d7a0'
    ],
    'Q': [
        '\u211a', '\u2d55', '\uff31', '\u1d410', '\u1d444', '\u1d478', '\u1d4ac', '\u1d4e0', '\u1d514',
        '\u1d57c', '\u1d5b0', '\u1d5e4', '\u1d618', '\u1d64c', '\u1d680'
    ],
    'R': [
        '\u01a6', '\u0280', '\u13a1', '\u13d2', '\u1587', '\u16b1', '\u211b', '\u211c', '\u211d', '\ua4e3',
        '\uab71', '\uaba2', '\uff32', '\u104b4', '\u16f35', '\u1d216', '\u1d411', '\u1d445', '\u1d479', '\u1d4e1',
        '\u1d57d', '\u1d5b1', '\u1d5e5', '\u1d619', '\u1d64d', '\u1d681'
    ],
    'S': [
        '\u0405', '\u054f', '\u13d5', '\u13da', '\ua4e2', '\uff33', '\u10296', '\u10420', '\u16f3a',
        '\u1d412', '\u1d446', '\u1d47a', '\u1d4ae', '\u1d4e2', '\u1d516', '\u1d54a', '\u1d57e', '\u1d5b2', '\u1d5e6',
        '\u1d61a', '\u1d64e', '\u1d682'
    ],
    'T': [
        '\u03a4', '\u03c4', '\u0422', '\u0442', '\u13a2', '\u1d1b', '\u22a4', '\u27d9', '\u2ca6', '\ua4d4',
        '\uab72', '\uff34', '\u10297', '\u102b1', '\u10315', '\u118bc', '\u16f0a', '\u1d413', '\u1d447', '\u1d47b',
        '\u1d4af', '\u1d4e3', '\u1d517', '\u1d54b', '\u1d57f', '\u1d5b3', '\u1d5e7', '\u1d61b', '\u1d64f', '\u1d683',
        '\u1d6bb', '\u1d6d5', '\u1d6f5', '\u1d70f', '\u1d72f', '\u1d749', '\u1d769', '\u1d783', '\u1d7a3', '\u1d7bd',
        '\u1f768'
    ],
    'U': [
        '\u054d', '\u1200', '\u144c', '\u222a', '\u22c3', '\ua4f4', '\uff35', '\u104ce', '\u118b8',
        '\u16f42', '\u1d414', '\u1d448', '\u1d47c', '\u1d4b0', '\u1d4e4', '\u1d518', '\u1d54c', '\u1d580', '\u1d5b4',
        '\u1d5e8', '\u1d61c', '\u1d650', '\u1d684'
    ],
    'V': [
        '\u0474', '\u0667', '\u06f7', '\u13d9', '\u142f', '\u2164', '\u2d38', '\ua4e6', '\ua6df', '\uff36',
        '\u1051d', '\u118a0', '\u16f08', '\u1d20d', '\u1d415', '\u1d449', '\u1d47d', '\u1d4b1', '\u1d4e5', '\u1d519',
        '\u1d54d', '\u1d581', '\u1d5b5', '\u1d5e9', '\u1d61d', '\u1d651', '\u1d685'
    ],
    'W': [
        '\u051c', '\u13b3', '\u13d4', '\ua4ea', '\uff37', '\u118e6', '\u118ef', '\u1d416', '\u1d44a',
        '\u1d47e', '\u1d4b2', '\u1d4e6', '\u1d51a', '\u1d54e', '\u1d582', '\u1d5b6', '\u1d5ea', '\u1d61e', '\u1d652',
        '\u1d686'
    ],
    'X': [
        '\u03a7', '\u0425', '\u166d', '\u16b7', '\u2169', '\u2573', '\u2cac', '\u2d5d', '\ua4eb', '\ua7b3',
        '\uff38', '\u10290', '\u102b4', '\u10317', '\u10322', '\u10527', '\u118ec', '\u1d417', '\u1d44b', '\u1d47f',
        '\u1d4b3', '\u1d4e7', '\u1d51b', '\u1d54f', '\u1d583', '\u1d5b7', '\u1d5eb', '\u1d61f', '\u1d653', '\u1d687',
        '\u1d6be', '\u1d6f8', '\u1d732', '\u1d76c', '\u1d7a6'
    ],
    'Y': [
        '\u03a5', '\u03d2', '\u0423', '\u04ae', '\u13a9', '\u13bd', '\u2ca8', '\ua4ec', '\uff39', '\u102b2',
        '\u118a4', '\u16f43', '\u1d418', '\u1d44c', '\u1d480', '\u1d4b4', '\u1d4e8', '\u1d51c', '\u1d550', '\u1d584',
        '\u1d5b8', '\u1d5ec', '\u1d620', '\u1d654', '\u1d688', '\u1d6bc', '\u1d6f6', '\u1d730', '\u1d76a', '\u1d7a4'
    ],
    'Z': [
        '\u0396', '\u13c3', '\u2124', '\u2128', '\ua4dc', '\uff3a', '\u102f5', '\u118a9', '\u118e5',
        '\u1d419', '\u1d44d', '\u1d481', '\u1d4b5', '\u1d4e9', '\u1d585', '\u1d5b9', '\u1d5ed', '\u1d621', '\u1d655',
        '\u1d689', '\u1d6ad', '\u1d6e7', '\u1d721', '\u1d75b', '\u1d795'
    ],
    'a': [
        '\u0251', '\u03b1', '\u0430', '\u237a', '\uff41', '\U0001d41a', '\U0001d44e', '\U0001d482', '\U0001d4b6',
        '\U0001d4ea', '\U0001d51e', '\U0001d552', '\U0001d586', '\U0001d5ba', '\U0001d5ee', '\U0001d622',
        '\U0001d656', '\U0001d68a', '\U0001d6c2', '\U0001d6fc', '\U0001d736', '\U0001d770', '\U0001d7aa'
    ],
    'b': [
        '\u0184', '\u042c', '\u13cf', '\u1472', '\u15af', '\uff42', '\U0001d41b', '\U0001d44f', '\U0001d483',
        '\U0001d4b7', '\U0001d4eb', '\U0001d51f', '\U0001d553', '\U0001d587', '\U0001d5bb', '\U0001d5ef',
        '\U0001d623', '\U0001d657', '\U0001d68b'
    ],
    'c': [
        '\u03f2', '\u0441', '\u1d04', '\u217d', '\u2ca5', '\uabaf', '\uff43', '\U0001043d', '\U0001d41c',
        '\U0001d450', '\U0001d484', '\U0001d4b8', '\U0001d4ec', '\U0001d520', '\U0001d554', '\U0001d588',
        '\U0001d5bc', '\U0001d5f0', '\U0001d624', '\U0001d658', '\U0001d68c'
    ],
    'd': [
        '\u0501', '\u13e7', '\u146f', '\u2146', '\u217e', '\ua4d2', '\uff44', '\U0001d41d', '\U0001d451',
        '\U0001d485', '\U0001d4b9', '\U0001d4ed', '\U0001d521', '\U0001d555', '\U0001d589', '\U0001d5bd',
        '\U0001d5f1', '\U0001d625', '\U0001d659', '\U0001d68d'
    ],
    'e': [
        '\u0435', '\u04bd', '\u212e', '\u212f', '\u2147', '\uab32', '\uff45', '\U0001d41e', '\U0001d452',
        '\U0001d486', '\U0001d4ee', '\U0001d522', '\U0001d556', '\U0001d58a', '\U0001d5be', '\U0001d5f2',
        '\U0001d626', '\U0001d65a', '\U0001d68e'
    ],
    'f': [
        '\u017f', '\u03dd', '\u0584', '\u1e9d', '\ua799', '\uab35', '\uff46', '\U0001d41f', '\U0001d453'
        '\U0001d487', '\U0001d4bb', '\U0001d4ef', '\U0001d523', '\U0001d557', '\U0001d58b', '\U0001d5bf',
        '\U0001d5f3', '\U0001d627', '\U0001d65b', '\U0001d68f', '\U0001d7cb'
    ],
    'g': [
        '\u018d', '\u0261', '\u0581', '\u1d83', '\u210a', '\uff47', '\U0001d420', '\U0001d454', '\U0001d488',
        '\U0001d4f0', '\U0001d524', '\U0001d558', '\U0001d58c', '\U0001d5c0', '\U0001d5f4', '\U0001d628',
        '\U0001d65c', '\U0001d690'],
    'h': [
        '\u04bb', '\u0570', '\u13c2', '\u210e', '\uff48', '\U0001d421', '\U0001d489', '\U0001d4bd', '\U0001d4f1',
        '\U0001d525', '\U0001d559', '\U0001d58d', '\U0001d5c1', '\U0001d5f5', '\U0001d629', '\U0001d65d',
        '\U0001d691'
    ],
    'j': [
        '\u03f3', '\u0458', '\u2149', '\uff4a', '\U0001d423', '\U0001d457', '\U0001d48b', '\U0001d4bf', '\U0001d4f3',
        '\U0001d527', '\U0001d55b', '\U0001d58f', '\U0001d5c3', '\U0001d5f7', '\U0001d62b', '\U0001d65f',
        '\U0001d693'
    ],
    'k': [
        '\U0001d424', '\U0001d458', '\U0001d48c', '\U0001d4c0', '\U0001d4f4', '\U0001d528', '\U0001d55c',
        '\U0001d590', '\U0001d5c4', '\U0001d5f8', '\U0001d62c', '\U0001d660', '\U0001d694'
    ],
    'm': [
        '\uff4d'
    ],
    'n': [
        '\u0578', '\u057c', '\uff4e', '\U0001d427', '\U0001d45b', '\U0001d48f', '\U0001d4c3', '\U0001d4f7',
        '\U0001d52b', '\U0001d55f', '\U0001d593', '\U0001d5c7', '\U0001d5fb', '\U0001d62f', '\U0001d663', '\U0001d697'
    ],
    'p': [
        '\u03c1', '\u03f1', '\u0440', '\u2374', '\u2ca3', '\uff50', '\U0001d429', '\U0001d45d', '\U0001d491',
        '\U0001d4c5', '\U0001d4f9', '\U0001d52d', '\U0001d561', '\U0001d595', '\U0001d5c9', '\U0001d5fd',
        '\U0001d631', '\U0001d665', '\U0001d699', '\U0001d6d2', '\U0001d6e0', '\U0001d70c', '\U0001d71a',
        '\U0001d746', '\U0001d754', '\U0001d780', '\U0001d78e', '\U0001d7ba', '\U0001d7c8'
    ],
    'q': [
        '\u051b', '\u0563', '\u0566', '\uff51', '\U0001d42a', '\U0001d45e', '\U0001d492', '\U0001d4c6', '\U0001d4fa',
        '\U0001d52e', '\U0001d562', '\U0001d596', '\U0001d5ca', '\U0001d5fe', '\U0001d632', '\U0001d666',
        '\U0001d69a'
    ],
    'r': [
        '\u0433', '\u1d26', '\u2c85', '\uab47', '\uab48', '\uab81', '\uff52', '\U0001d42b', '\U0001d45f',
        '\U0001d493', '\U0001d4c7', '\U0001d4fb', '\U0001d52f', '\U0001d563', '\U0001d597', '\U0001d5cb',
        '\U0001d5ff', '\U0001d633', '\U0001d667', '\U0001d69b'
    ],
    's': [
        '\u01bd', '\u0455', '\ua731', '\uabaa', '\uff53', '\U00010448', '\U000118c1', '\U0001d42c', '\U0001d460',
        '\U0001d494', '\U0001d4c8', '\U0001d4fc', '\U0001d530', '\U0001d564', '\U0001d598', '\U0001d5cc',
        '\U0001d600', '\U0001d634', '\U0001d668', '\U0001d69c'
    ],
    't': [
        '\uff54', '\U0001d42d', '\U0001d461', '\U0001d495', '\U0001d4c9', '\U0001d4fd', '\U0001d531', '\U0001d565',
        '\U0001d599', '\U0001d5cd', '\U0001d601', '\U0001d635', '\U0001d669', '\U0001d69d'
    ],
    'u': [
        '\u028b', '\u03c5', '\u057d', '\u1d1c', '\ua79f', '\uab4e', '\uab52', '\uff55', '\U000104f6',
        '\U000118d8', '\U0001d42e', '\U0001d462', '\U0001d496', '\U0001d4ca', '\U0001d4fe', '\U0001d532',
        '\U0001d566', '\U0001d59a', '\U0001d5ce', '\U0001d602', '\U0001d636', '\U0001d66a', '\U0001d69e',
        '\U0001d6d6', '\U0001d710', '\U0001d74a', '\U0001d784', '\U0001d7be'
    ],
    'v': [
        '\u03bd', '\u0475', '\u05d8', '\u1d20', '\u2174', '\u2228', '\u22c1', '\uaba9', '\uff56', '\U00011706',
        '\U000118c0', '\U0001d42f', '\U0001d463', '\U0001d497', '\U0001d4cb', '\U0001d4ff', '\U0001d533',
        '\U0001d567', '\U0001d59b', '\U0001d5cf', '\U0001d603', '\U0001d637', '\U0001d66b', '\U0001d69f',
        '\U0001d6ce', '\U0001d708', '\U0001d742', '\U0001d77c', '\U0001d7b6'
    ],
    'w': [
        '\u026f', '\u0461', '\u051d', '\u0561', '\u1d21', '\uab83', '\uff57', '\U0001170a', '\U0001170e',
        '\U0001170f', '\U0001d430', '\U0001d464', '\U0001d498', '\U0001d4cc', '\U0001d500', '\U0001d534',
        '\U0001d568', '\U0001d59c', '\U0001d5d0', '\U0001d604', '\U0001d638', '\U0001d66c', '\U0001d6a0'
    ],
    'x': [
        '\u00d7', '\u0445', '\u1541', '\u157d', '\u166e', '\u2179', '\u292b', '\u292c', '\u2a2f', '\uff58',
        '\U0001d431', '\U0001d465', '\U0001d499', '\U0001d4cd', '\U0001d501', '\U0001d535', '\U0001d569',
        '\U0001d59d', '\U0001d5d1', '\U0001d605', '\U0001d639', '\U0001d66d', '\U0001d6a1'
    ],
    'y': [
        '\u0263', '\u028f', '\u03b3', '\u0443', '\u04af', '\u10e7', '\u1d8c', '\u1eff', '\u213d', '\uab5a',
        '\uff59', '\U000118dc', '\U0001d432', '\U0001d466', '\U0001d49a', '\U0001d4ce', '\U0001d502', '\U0001d536',
        '\U0001d56a', '\U0001d59e', '\U0001d5d2', '\U0001d606', '\U0001d63a', '\U0001d66e', '\U0001d6a2',
        '\U0001d6c4', '\U0001d6fe', '\U0001d738', '\U0001d772', '\U0001d7ac'
    ],
    'z': [
        '\u01d22', '\uab93', '\uff5a', '\U000118c4', '\U0001d433', '\U0001d467', '\U0001d49b', '\U0001d4cf',
        '\U0001d503', '\U0001d537', '\U0001d56b', '\U0001d59f', '\U0001d5d3', '\U0001d607', '\U0001d63b', '\U0001d66f', '\U0001d6a3'
    ]
}

# generating reverse homoglyph dictionary using homoglyph_mapping
def generate_dict(homoglyph_mapping, filename = 'reverse_mapping.json'):
    reverse_mapping = {glyph: char for char, glyphs in homoglyph_mapping.items() for glyph in glyphs}
    with open(filename, 'w') as file:
       json.dump(reverse_mapping, file, indent=4)

def replace_homoglyphs(text, reverse_mapping):
    with open(reverse_mapping, 'r') as file:
        data = json.load(file)
    corrected_text = ''.join(data.get(char, char) for char in text)
    return corrected_text

# inherit function for fixed seed
class FixSeedAugmenter(Augmenter):
    def __init__(
        self,
        transformation,
        constraints=[],
        pct_words_to_swap=0.1,
        transformations_per_example=1,
        high_yield=False,
        fast_augment=False,
        enable_advanced_metrics=False,
    ):
        super().__init__(
            transformation,
            constraints,
            pct_words_to_swap,
            transformations_per_example,
            high_yield,
            fast_augment,
            enable_advanced_metrics
        )

    def _filter_transformations(self, transformed_texts, current_text, original_text, seed=54):
        set_seed(seed) # fixing seed
        transformed_texts = super()._filter_transformations(transformed_texts,current_text,original_text)
        return transformed_texts

# inherit function for translation from english to spanish
class translation(BackTranslation):
    def __init__(
        self,
        src_lang="en",
        target_lang="es",
        src_model="Helsinki-NLP/opus-mt-ROMANCE-en",
        target_model="Helsinki-NLP/opus-mt-en-ROMANCE",
        chained_back_translation=0,
    ):
        super().__init__(src_lang, target_lang, src_model, target_model, chained_back_translation)

    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []
        current_text = current_text.text

        # to perform chained back translation, a random list of target languages are selected from the provided model
        if self.chained_back_translation:
            list_of_target_lang = random.sample(
                self.target_tokenizer.supported_language_codes,
                self.chained_back_translation,
            )
            for target_lang in list_of_target_lang:
                target_language_text = self.translate(
                    [current_text],
                    self.target_model,
                    self.target_tokenizer,
                    target_lang,
                )
                current_text = target_language_text[0]
            return [AttackedText(current_text)]

        # translates source to target language and back to source language (single back translation)
        target_language_text = self.translate(
            [current_text], self.target_model, self.target_tokenizer, self.target_lang
        )
        transformed_texts.append(AttackedText(target_language_text[0]))
        return transformed_texts

# Synonym Swap
def embedding_augment(input_text, pct_words_to_swap=0.5, transformations_per_example=20, max_mse_dist=0.2):
    # Set up transformation
    transformation = WordSwapEmbedding()
    # Set up constraints
    constraints = [RepeatModification(), StopwordModification(), WordEmbeddingDistance(max_mse_dist=max_mse_dist)]
    # Create augmenter with specified parameters
    augmenter = FixSeedAugmenter(
        transformation=transformation,
        constraints=constraints,
        pct_words_to_swap=pct_words_to_swap,
        transformations_per_example=transformations_per_example,
    )
    # Perform augmentation
    result = augmenter.augment(input_text)
    return result

# translate from english to spanish
def translation_augment(input_text, pct_words_to_swap=0.5, transformations_per_example=1):
    transformation = translation()
    constraints = [RepeatModification(), StopwordModification(), ]

    augmenter = FixSeedAugmenter(
        transformation=transformation,
        constraints=constraints,
        pct_words_to_swap=pct_words_to_swap,
        transformations_per_example=transformations_per_example,
    )

    result = augmenter.augment(input_text)
    return result

# delete or replace characters
def morphing_augment(input_text, pct_words_to_swap=1, transformations_per_example=10, max_mse_dist=0.01):
    transformation = CompositeTransformation(
        [WordSwapRandomCharacterDeletion(), WordSwapQWERTY()]
    )
    constraints = [RepeatModification(), StopwordModification(), WordEmbeddingDistance(max_mse_dist=max_mse_dist)]

    augmenter = FixSeedAugmenter(
        transformation=transformation,
        constraints=constraints,
        pct_words_to_swap=pct_words_to_swap,
        transformations_per_example=transformations_per_example,
    )

    result = augmenter.augment(input_text)
    return result


# example
if __name__ == "__main__":

    # Perturbation example
    s = "beautiful car"
    augmented_texts = embedding_augment(s)
    # augmented_texts = morphing_augment(s)
    # augmented_texts = translation_augment(s)
    for line in augmented_texts:
        print(f"a photo of an {line}")
    print(f"{len(augmented_texts)} samples")

    # replace homoglyphs example
    text_with_homoglyphs = "toଠl, abcauodociaz"
    generate_dict(homoglyph_mapping, 'reverse_mapping.json')
    text = replace_homoglyphs(text_with_homoglyphs, 'reverse_mapping.json')
    print(f"Corrected text: {text}")
