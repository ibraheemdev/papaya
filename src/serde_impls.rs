use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use std::fmt::{self, Formatter};
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

use crate::{Guard, HashMap, HashMapRef};

struct MapVisitor<K, V, S> {
    _marker: PhantomData<HashMap<K, V, S>>,
}

impl<K, V, S, G> Serialize for HashMapRef<'_, K, V, S, G>
where
    K: Serialize + Hash + Eq,
    V: Serialize,
    G: Guard,
    S: BuildHasher,
{
    fn serialize<Sr>(&self, serializer: Sr) -> Result<Sr::Ok, Sr::Error>
    where
        Sr: Serializer,
    {
        serializer.collect_map(self)
    }
}

impl<K, V, S> Serialize for HashMap<K, V, S>
where
    K: Serialize + Hash + Eq,
    V: Serialize,
    S: BuildHasher,
{
    fn serialize<Sr>(&self, serializer: Sr) -> Result<Sr::Ok, Sr::Error>
    where
        Sr: Serializer,
    {
        self.pin().serialize(serializer)
    }
}

impl<'de, K, V, S> Deserialize<'de> for HashMap<K, V, S>
where
    K: Deserialize<'de> + Hash + Eq,
    V: Deserialize<'de>,
    S: Default + BuildHasher,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(MapVisitor::new())
    }
}

impl<K, V, S> MapVisitor<K, V, S> {
    pub(crate) fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<'de, K, V, S> Visitor<'de> for MapVisitor<K, V, S>
where
    K: Deserialize<'de> + Hash + Eq,
    V: Deserialize<'de>,
    S: Default + BuildHasher,
{
    type Value = HashMap<K, V, S>;

    fn expecting(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "a map")
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let values = match access.size_hint() {
            Some(size) => HashMap::with_capacity_and_hasher(size, S::default()),
            None => HashMap::default(),
        };

        {
            let values = values.pin();
            while let Some((key, value)) = access.next_entry()? {
                values.insert(key, value);
            }
        }

        Ok(values)
    }
}

#[cfg(test)]
mod test {
    use crate::HashMap;

    #[test]
    fn test_map() {
        let map: HashMap<u8, u8> = HashMap::new();
        let guard = map.guard();

        map.insert(0, 4, &guard);
        map.insert(1, 3, &guard);
        map.insert(2, 2, &guard);
        map.insert(3, 1, &guard);
        map.insert(4, 0, &guard);

        let serialized = serde_json::to_string(&map).unwrap();
        let deserialized = serde_json::from_str(&serialized).unwrap();

        assert_eq!(map, deserialized);
    }
}
