// VisitorMap.tsx
import { Flex, Center } from '@chakra-ui/react';
import React, { useEffect, useRef } from 'react';

const VisitorMap: React.FC = () => {
  const mapRef = useRef<HTMLDivElement | null>(null);  // Specify the type of ref
  
  useEffect(() => {
    const script = document.createElement('script');
    script.type = 'text/javascript';
    script.src = "//clustrmaps.com/globe.js?d=TUO8N9WIGh-Bp3p30oWZl2UqJVaob59wyMNfbhxLSi8";
    script.id = "clstr_globe";
    if (mapRef.current) {
      mapRef.current.appendChild(script);  // Append the script to the div element
    }

    // Optional: Cleanup the script on component unmount
    return () => {
      if (mapRef.current) {
        mapRef.current.removeChild(script);
      }
    };
  }, []);  // Empty dependency array ensures this useEffect runs once

  return (
    <Center>
      <Flex ref={mapRef} w={'100px'} h={'100px'} mx={'auto'} alignItems={'center'} justify={'center'} mt={8}></Flex>
    </Center>
  );
};

export default VisitorMap;
