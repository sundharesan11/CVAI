window.onscroll = function() {
    var sections = document.querySelectorAll("section");
    var navigation_links = document.querySelectorAll(".navigation-link");
    var currentSectionId = "";

    sections.forEach(function(section) {
        var sectionTop = section.offsetTop - 100;
        var sectionHeight = section.clientHeight;
        if (pageYOffset >= sectionTop && pageYOffset < sectionTop + sectionHeight) {
            currentSectionId = section.getAttribute("id");
        }
    });

    navigation_links.forEach(function(link) {
        link.classList.remove("active");
        if (link.getAttribute("href").substring(1) === currentSectionId) {
            link.classList.add("active");
        }
    });
};

function smoothScrollTo(sectionId) {
    var element = document.getElementById(sectionId);
    element.scrollIntoView({ behavior: "smooth" });
}
