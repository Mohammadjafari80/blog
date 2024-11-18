import {
    HStack,
    Menu,
    MenuButton,
    MenuList,
    MenuItem,
    IconButton,
    useBreakpointValue,
    Portal,
    Spacer,
    useColorMode,
    Tabs,
    TabList,
    Tab,
    TabIndicator,
  } from "@chakra-ui/react";
  import { HamburgerIcon } from "@chakra-ui/icons";
  import ColorModeSwitch from "./ColorModeSwitch";
  import { Link } from "react-scroll"; // Import Link from react-scroll
  
  const NavBar = () => {
    const padding = 10;
    const isMobile = useBreakpointValue({ base: true, md: false });
    const { colorMode } = useColorMode();
    const bgColor = colorMode === "dark" ? "#121212" : "#FFFFFF";
    
    const renderMenuItems = () => (
      <Tabs variant="unstyled">
        <TabList>
          {[
                { name: 'Blog', path: '/blogs' },
                { name: 'Main Site', path: 'https://mohammadjafari80.github.io' }, // Replace with your main site URL
            ].map((item) => (
            <Link key={item.name} smooth={true} duration={500}>
              <Tab fontSize={15} fontWeight="bold" aria-label={`${item.name} section`}>
                {item.name}
              </Tab>
            </Link>
          ))}
        </TabList>
        <TabIndicator mt="-40px" height="2px" bg="#d05a45" borderRadius="5px" />
      </Tabs>
    );
  
    const renderDropdownMenu = () => (
      <Menu>
        <MenuButton as={IconButton} icon={<HamburgerIcon />} mr={5} aria-label="Navigation menu" />
        <Portal>
          <MenuList width="100%" position="fixed" top="0" left="0" bg={bgColor}>
            {[
                { name: 'Blog', path: '/blogs' },
                { name: 'Main Site', path: 'https://mohammadjafari80.github.io' }, // Replace with your main site URL
            ].map((item) => (
              <Link key={item.name} smooth={true} duration={500}>
                <MenuItem
                  bg={bgColor}
                  onClick={() => {
                    const burger = document.getElementById("burger");
                    if (burger) {
                      burger.click();
                    }
                  }}
                >
                  {item.name}
                </MenuItem>
              </Link>
            ))}
            <MenuItem bg={bgColor}>
              <ColorModeSwitch />
            </MenuItem>
          </MenuList>
        </Portal>
      </Menu>
    );
  
    return (
      <HStack
        m={padding}
        bgColor={bgColor}
        alignItems="baseline"
        w="100%"
        pl="15px"
        pr="15px"
        ml="auto"
        mr="auto"
        mb="0px"
        maxWidth="95%"
        css={{
          "@media (min-width: 1200px)": {
            maxWidth: "60%",
          },
        }}
      >
        {isMobile ? renderDropdownMenu() : renderMenuItems()}
        {!isMobile && (
          <>
            <Spacer />
            <ColorModeSwitch />
          </>
        )}
      </HStack>
    );
  };
  
  export default NavBar;
  