import string
import re
from typing import List, Text

TEXT = """{Description of ""External Tank"" option for SSF redesign deleted}


Yo Ken, let's keep on-top of things! Both the ""External Tank"" and
""Wingless Orbiter"" options have been deleted from the SSF redesign
options list. Today's (4/23) edition of the New York Times reports
that O'Connor told the panel that some redesign proposals have
been dropped, such as using the ""giant external fuel tanks used
in launching space shuttles,"" and building a ""station around
an existing space shuttle with its wings and tail removed.""

Currently, there are three options being considered, as presented
to the advisory panel meeting yesterday (and as reported in
today's Times).

Option ""A"" - Low Cost Modular Approach
This option is being studied by a team from MSFC. {As an aside,
there are SSF redesign teams at MSFC, JSC, and LaRC supporting
the SRT (Station Redesign Team) in Crystal City. Both LeRC and
Reston folks are also on-site at these locations, helping the respective
teams with their redesign activities.} Key features of this
option are:
  -  Uses ""Bus-1"", a modular bus developed by Lockheed that's
     qualified for STS and ELV's. The bus provides propulsion, GN&C
     Communications, & Data Management. Lockheed developed this
     for the Air Force.
  -  A ""Power Station Capability"" is obtained in 3 Shuttle Flights.
     SSF Solar arrays are used to provide 20 kW of power. The vehicle
     flies in an ""arrow mode"" to optimize the microgravity environment.
     Shuttle/Spacelab missions would utilize the vehilce as a power
     source for 30 day missions.
  -  Human tended capability (as opposed to the old SSF sexist term
     of man-tended capability) is achieved by the addition of the
     US Common module. This is a modified version of the existing
     SSF Lab module (docking ports are added for the International
     Partners' labs, taking the place of the nodes on SSF). The
     Shuttle can be docked to the station for 60 day missions.
     The Orbiter would provide crew habitability & EVA capability.
  -  International Human Tended. Add the NASDA & ESA modules, and
     add another 20 kW of power
  -  Permanent Human Presence Capability. Add a 3rd power module,
     the U.S. habitation module, and an ACRV (Assured Crew Return
     Vehicle).

Option ""B"" - Space Station Freedom Derived
The Option ""B"" team is based at LaRC, and is lead by Mike Griffin.
This option looks alot like the existing SSF design, which we
have all come to know and love :)

This option assumes a lightweight external tank is available for
use on all SSF assembly flights (so does option ""A""). Also, the 
number of flights is computed for a 51.6 inclination orbit,
for both options ""A"" and ""B"".

The build-up occurs in six phases:
  -  Initial Research Capability reached after 3 flights. Power
     is transferred from the vehicle to the Orbiter/Spacelab, when
     it visits.
  -  Man-Tended Capability (Griffin has not yet adopted non-sexist
     language) is achieved after 8 flights. The U.S. Lab is
     deployed, and 1 solar power module provides 20 kW of power.
  -  Permanent Human Presence Capability occurs after 10 flights, by
     keeping one Orbiter on-orbit to use as an ACRV (so sometimes
     there would be two Orbiters on-orbit - the ACRV, and the
     second one that comes up for Logistics & Re-supply).
  -  A ""Two Fault Tolerance Capability"" is achieved after 14 flights,
     with the addition of a 2nd power module, another thermal
     control system radiator, and more propulsion modules.
  -  After 20 flights, the Internationals are on-board. More power,
     the Habitation module, and an ACRV are added to finish the
     assembly in 24 flights.

Most of the systems currently on SSF are used as-is in this option, 
with the exception of the data management system, which has major
changes.
"""

def split_sentences(
    text: Text
)-> List[Text]:
    """
    Splits a text string into a list of individual sentences.

    This function uses a regular expression to identify sentence boundaries based
    on punctuation and common abbreviations. It handles edge cases like
    sentences ending in abbreviations or questions marks.

    Args:
        text (str): The text string to split into sentences.

    Returns:
        list: A list of strings, where each string is a single sentence.

    Examples:
        >>> split_sentences("This is a sentence. Is this another? It is!")
        ['This is a sentence.', 'Is this another?', 'It is!']

        >>> split_sentences("I love N.Y. It's a great city.")
        ['I love N.Y.', "It's a great city."]

        >>> split_sentences("Dr. Smith said, 'Hello there!'")
        ['Dr', "Smith said, 'Hello there!'"]
    """
    text = text = re.sub("\n"," ",text)
    text = re.sub(r"[!:\.\?\-;]\s",r"__",text)
    text = re.sub("[\t\n\r\x0b\x0c]", '__', text)
    # text = re.sub(r"\"\"", '__', text)
    text = re.sub(r'\s\s+', '__', text)
    sents = text.split('__')
    return sents

if __name__ == "__main__":
    print(split_sentences(TEXT))
    