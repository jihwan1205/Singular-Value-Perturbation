$(document).ready(function() {
    // Mobile navbar toggle. Keyboard-operable and state-announcing: below 1024px
    // this burger is the only navigation on the page.
    $(".navbar-burger").on("click keydown", function(e) {
      if (e.type === "keydown" && e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      var open = $(this).toggleClass("is-active").hasClass("is-active");
      $(".navbar-menu").toggleClass("is-active", open);
      $(this).attr("aria-expanded", open);
    });

    // Initialize any carousel on the page (none at the moment; harmless if absent).
    bulmaCarousel.attach('.carousel', {
      slidesToScroll: 1,
      slidesToShow: 3,
      loop: true,
      infinite: true,
      autoplay: false,
    });

    bulmaSlider.attach();
})
